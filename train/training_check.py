import pandas as pd
import os
import json
import numpy as np

def print_job_array(jobs):
    print('[', end='')
    print(*jobs, sep=',', end='')
    print(']')

def run(exps, verbose=False):
    jobs = []

    for exp in exps.itertuples():
        if not os.path.exists(exp.stats):
            jobs.append(exp.job_id)
            if verbose:
                print(exp.job_id, 'No stat file')
                print(exp.model_type)
            continue
        print(exp.job_id)
        d = pd.read_csv(exp.stats)
        if len(d) < exp.max_epoch:
            jobs.append(exp.job_id)
            if verbose:
                print(exp.job_id, len(d), exp.num)

    return jobs

if __name__ == '__main__':
    exps = pd.read_csv('/home/avic/Rotation-Generalization/exps_half.csv')
    exps = exps[exps.model_type.isin(['ModelType.ResNet', 'ModelType.DenseNet', 'ModelType.Inception', 'ModelType.CorNet'])]
    exps = exps[np.sum(np.array([(exps.augment).to_numpy(dtype=np.int), (exps.pretrained).to_numpy(dtype=np.int), (exps.scale).to_numpy(dtype=np.int)]), axis=0) <= 1]
    # exps = exps[exps.pretrained & ~exps.scale & ~exps.augment]
    # exps = exps[exps.training_category == "SM"]

    print(len(exps))
    jobs = run(exps, verbose=True)

    print(f'All jobs:')
    print(f'Array format: 0-{len(jobs) - 1}')
    print_job_array(jobs)

    with open('remaining_jobs.json', 'w') as outfile:
        json.dump(jobs, outfile)
