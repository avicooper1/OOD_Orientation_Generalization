import numpy as np
from PIL import Image
import os
from tqdm.contrib.concurrent import process_map
import sys
import pandas as pd
sys.path.append('/home/avic/OOD_Orientation_Generalization')
from my_dataclasses import DatasetPath

def compress_image(img_path):
    img = np.array(Image.open(img_path))
    img[np.where(img <= 1)] = 0
    Image.fromarray(img).save(img_path, compress_level=9)

if __name__ == '__main__':
    jobs = []
    for dataset_i in range(4):
        dataset_path = DatasetPath.get_dataset(sys.argv[-2], dataset_i, int(sys.argv[-1]))
        dataset_path.load()
        if not dataset_path.complete:
            dataset_complete = True
            for subdataset_i in range(50):
                current_cubelet = dataset_path.check_annotation_file(subdataset_i)
                if current_cubelet != -1:
                    print(current_cubelet)
                    dataset_complete = False
                    jobs.append((dataset_i * 50) + subdataset_i)
            if dataset_complete:
                pd.concat([dataset_path.annotation_file(i) for i in range(50)]).reset_index(drop=True).to_csv(
                    dataset_path.merged_annotation_path(), index=False)
                dataset_path.complete = True
                dataset_path.save()
        if dataset_path.complete and not dataset_path.images_compressed:
            process_map(compress_image, map(lambda f: os.path.join(dataset_path.image_dir, f), os.listdir(dataset_path.image_dir)))
            dataset_path.images_compressed = True
            dataset_path.save()

    print('[', end='')
    print(*jobs, sep=',', end='')
    print(']')