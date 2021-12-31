import pandas as pd
import sys
sys.path.append('/home/avic/Rotation-Generalization')
from dataset_path import DatasetPath
import tqdm

for i in tqdm.tqdm(range(0, 40)):
    dataset_path = DatasetPath.get_dataset(i)
    print(dataset_path)
    pd.concat([pd.read_csv(p) for p in dataset_path.annotations_paths()]).to_csv(dataset_path.merged_annotation_path())