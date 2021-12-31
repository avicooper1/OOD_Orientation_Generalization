import pandas as pd
import os
import sys
sys.path.append('/home/avic/Rotation-Generalization')
from dataset_path import DatasetPath, DatasetType

jobs = []

for i in range(0, 2000):

    dataset_path = DatasetPath.get_dataset(i // 50)

    if not os.path.exists(dataset_path.path):
        print(f'{i}\tError: dataset at {dataset_path.path} does not exist')
        jobs.append(i)
        continue

    elif not os.path.exists(dataset_path.annotation_path_i(i % 50)):
        print(f'{i}\tError: dataset annotations file at {dataset_path.annotation_path_i(i % 50)} does not exist')
        jobs.append(i)
        continue

    try:
        d = pd.read_csv(dataset_path.annotation_path_i(i % 50))
        if len(d) < (5_000 if dataset_path.type == DatasetType.Full else 4_000):
            print(dataset_path.path)
            if dataset_path.type == DatasetType.Full:
                print(f'{i}\tCategory: {dataset_path.model_category}\tType: {dataset_path.type}\tScale: {dataset_path.scale}:\t{len(d)}')
            else:
                print(
                    f'{i}\tCategory: {dataset_path.model_category}\tType: {dataset_path.type}\tScale: {dataset_path.scale}\tAxes: {dataset_path.restriction_axes_named}\t{len(d)}')
            # if len(d) <= 2:
            #     print("deleting dataset")
            #     os.remove(dataset_path.annotation_path_i(i % 50))
            jobs.append(i)
    except Exception as e:
        print(e)
        print(f'Error at dataset {i, dataset_path.annotation_path_i(i % 50)}')

print('[', end='')
print(*jobs, sep=',', end='')
print(']')
