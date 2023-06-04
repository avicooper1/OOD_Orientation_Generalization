import argparse
import sys
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
import torch
from itertools import chain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check training status and generate results')
    parser.add_argument('project_path', help='path to the project source directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('--nums',
                        type=int,
                        nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                                 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 
                                 47, 48, 49, 55, 56, 57, 58],
                        help='experiment numbers')
    parser.add_argument('-fm', '--fit_model', nargs='?', default=False, const=True)
    parser.add_argument('-ra', '--run_analysis', nargs='?', default=False, const=True)
    parser.add_argument('-ow', '--overwrite_results', nargs='?', default=False, const=True)
    parser.add_argument('-v', '--verbose', nargs='?', default=False, const=True)
    args = parser.parse_args()
    
    sys.path.insert(0, args.project_path)
    
    from predictive_function.tools import sigmoid_cp, pf_func_path, fit_model
    from utils.persistent_data_class import ExpData
    from analysis.result import Result
    
    jobs_per_num = 4 * 5  # Number of num_fully_seen times number of runs
    job_list = list(chain(*[range(jobs_per_num * n, (jobs_per_num * (n + 1))) for n in args.nums]))
    
    remaining_train = []
    remaining_activations = []
    remaining_fit_nums = []
    remaining_fits = {}
    remaining_analysis_nums = []
    remaining_analysis = []

    if args.fit_model:
        torch.multiprocessing.set_start_method('spawn')

    if args.overwrite_results:
        if args.fit_model:
            response = input('About to delete fitted parameters. These can quite long to regenerate. Are you sure you want to continue? Type YES to continue: ')
            if response != 'YES':
                exit()

    
    for job_i in tqdm(job_list, desc='Checking analysis statuses', file=sys.stdout):

        try:
            exp_data = ExpData.from_job_i(args.project_path, args.storage_path, job_i)
        except (AssertionError, UnboundLocalError):
            if args.verbose:
                print(f'Experiment with id: {job_i} has not been run yet.')
            remaining_train.append(job_i)
            continue

        if not exp_data.complete:
            if args.verbose:
                print(f'Exp with job_i: {job_i}\n and dir {exp_data.dir}\n did not finish training. It trained for {len(exp_data.eval_data.partial_base_accuracies)} epochs, however it is meant to train for {exp_data.min_epochs}, and the last 7 accuracies were {exp_data.eval_data.partial_ood_accuracies[-7:]}')
            remaining_train.append(job_i)
            continue

        if sum([not a.on_disk() for a in exp_data.eval_data.activations + exp_data.eval_data.corrects]) > 0:
            if args.verbose:
                print(f'Exp with job_i: {job_i} completed but did not save activations to disk')
            remaining_activations.append(job_i)
            continue

        result = Result.from_exp_data(exp_data, args.project_path)

        if args.overwrite_results:
            if args.fit_model:
                result.predictive_model_params = []
            if args.run_analysis:
                result.full_id_acc = None
            result.save()
            continue

        if not result.predictive_model_params:
            remaining_fit_nums.append(result.num)
            if result.free_axis in remaining_fits:
                remaining_fits[result.free_axis].append(result)
            else:
                remaining_fits[result.free_axis] = [result]
            continue

        if result.full_id_acc is None:
            remaining_analysis_nums.append(result.num)
            remaining_analysis.append(result)
            continue

    print(f'Jobs that still need training: [{",".join(map(str, remaining_train))}]')
    print(f'Jobs that have completed training but lack activations: [{",".join(map(str, remaining_activations))}]')
    print(f'Jobs to be submitted to training script (set uninon of above): [{",".join(map(str, sorted(set(remaining_train + remaining_activations))))}]')
    print(f'Nums that have jobs that need fitting: [{",".join(map(str, sorted(set(remaining_fit_nums))))}]')
    print(f'Nums that have jobs that need analysis: [{",".join(map(str, sorted(set(remaining_analysis_nums))))}]')
            
    if args.fit_model and len(remaining_fits.keys()) > 0:

        for free_axis in remaining_fits:

            print(f'Fitting experiments for {free_axis} model')

            predictive_model = torch.tensor(np.load(pf_func_path(free_axis, args.project_path)).reshape(4, -1)).cuda()

            num_runs = 5
            num_epochs = 10_000

            with tqdm(total=len(remaining_fits[free_axis]) * 4 * num_runs,
                      desc='Fitting',
                      file=sys.stdout) as pbar:
                for result in remaining_fits[free_axis]:
                    fit_model(result, predictive_model, pbar)


    def run_analysis(r):
        r.run_analysis()

    if args.run_analysis and len(remaining_analysis) > 0:
        np.seterr(all='raise')
        # process_map(run_analysis, remaining_analysis, max_workers=16)
        with tqdm(total=len(remaining_analysis), desc='Running analysis', file=sys.stdout) as pbar:
            for _ in map(run_analysis, remaining_analysis):
                pbar.update(1)
            
