import os
import shutil
import argparse
import sys
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from numpy import argmax
import numpy as np


def process_job_i(a):
    job_i, args = a
    try:
        exp_data = ExpData.from_job_i(args.project_path, args.storage_path, job_i)
    except AssertionError:
        print(f'Experiment with id: {job_i} has not been run yet.')
        return job_i

    if args.activation_check:
        if exp_data.complete and not exp_data.eval_data.full_validation_activations.on_disk():
            print(f'Exp with job_i: {job_i} completed but did not save activations to disk')
            return job_i
    
    if not exp_data.complete:
        print(f'Exp with job_i: {job_i}\n and dir {exp_data.dir}\n did not finish training. It trained for {len(exp_data.eval_data.partial_base_accuracies)} epochs, however it is meant to train for {exp_data.min_epochs}, and the last 7 accuracies were {exp_data.eval_data.partial_ood_accuracies[-7:]}')
        return job_i

    if args.train_check:
        return
    
    if (not args.overwrite_results) and os.path.exists(Result.results_dir(exp_data)[0]):
        return

    if (args.overwrite_results or args.delete_results) and os.path.exists(Result.results_dir(exp_data)[0]):
        shutil.rmtree(Result.results_dir(exp_data)[0])
        if args.delete_results:
            return
    Result.from_job_i(args.project_path, args.storage_path, job_i)
    # except:
    #     print(f'Generating result with job_i {job_i} failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check training status and generate results')
    parser.add_argument('project_path', help='path to the project source directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('--nums',
                        type=int,
                        nargs='*',
                        help="experiment numbers")
    parser.add_argument('--job_i',
                        type=int,
                        nargs='?',
                        default=None,
                        help="job i to process")
    parser.add_argument('-tc', '--train_check', nargs='?', default=False, const=True)
    parser.add_argument('-ac', '--activation_check', nargs='?', default=False, const=True)
    parser.add_argument('-dr', '--delete_results', nargs='?', default=False, const=True)
    parser.add_argument('-ow', '--overwrite_results', nargs='?', default=False, const=True)
    args = parser.parse_args()
    
    sys.path.insert(0, args.project_path)
    
    from utils.persistent_data_class import ExpData
    from analysis.result import Result

    assert (args.job_i is not None) ^ (len(args.nums) > 0), 'Either a list of job or experiment numbers must be provided'
    
    jobs_per_num = 4 * 5  # Number of num_fully_seen times number of runs
    np.seterr('raise')
    if args.job_i is not None:
        process_job_i((args.job_i, args))
    else:
        remaining_jobs = []
        with Pool(32) as pool:
            for job_i in tqdm(pool.imap_unordered(process_job_i, ((job_i, args) for exp_num in args.nums for job_i in range(exp_num * jobs_per_num, (exp_num + 1) * jobs_per_num))), total=len(args.nums) * jobs_per_num):
                if job_i is not None:
                    remaining_jobs.append(job_i)
        print(remaining_jobs)
            