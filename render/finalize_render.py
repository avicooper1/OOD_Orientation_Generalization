import numpy as np
from PIL import Image
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import sys
sys.path.append('/home/avic/OOD_Orientation_Generalization')
from utils.persistent_data_class import ImageDataset

def i_from_name(name):
    return int(name[5:-4])

def check_image(args):
    path, name, _ = args

    image_exists = os.path.exists(path)
    i = i_from_name(name)

    image_ok = True
    if image_exists:
        if os.stat(path).st_size == 0:
            os.remove(path)
            image_ok = False
    else:
        image_ok = False

    return i, image_ok

def compress_image(args):
    path = args[0]

    img, _ = npImage(args, crop=False)
    img[np.where(img <= 1)] = 0
    Image.fromarray(img).save(path, compress_level=9)

def npImage(args, crop=True):
    path, name, program_args = args

    img = np.array(Image.open(path))
    if crop and program_args.crop > 0:
        img = img[program_args.crop:-program_args.crop, program_args.crop:-program_args.crop]
    return img, i_from_name(name)

def pool_over_imgs(image_dir, apply_func, msg, program_args, on_callback=None, callback_args=None, num_workers=8):
    with Pool(num_workers) as p:
        for result in tqdm(p.imap_unordered(apply_func,
                                            ((f.path, f.name, program_args) for f in os.scandir(image_dir))),
                           desc=msg,
                           total=args.resolution ** 3):
            if on_callback is not None:
                if callback_args is None:
                    on_callback(result)
                else:
                    on_callback(result, callback_args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check, compress and group images')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('resolution', type=int, help="dataset resolution")
    parser.add_argument('datasets', nargs='+', type=str, help="datasets to process", default=['plane',
                                                                                                      'car',
                                                                                                      'lamp,'
                                                                                                      'SM'])
    parser.add_argument('-sd', '--sub_datasets', nargs='+', type=int, help="sub-datasets to process",
                        default=range(50))
    parser.add_argument('-c', '--crop', type=int, help="border crop for grouping images",
                        default=15)
    parser.add_argument('-rc', '--render_check', nargs='?', default=False, const=True)

    args = parser.parse_args()

    for dataset_category in args.datasets:
        dataset = ImageDataset(args.storage_path, dataset_category, args.resolution)
        for subdataset_i in args.sub_datasets:
            subdataset = dataset.subdatasets[subdataset_i]
            
            if not subdataset.rendering_complete:
                dataset_complete = [True, 0]

                def callback(args, dataset_complete_for_callback):
                    i, image_ok = args
                    dataset_complete_for_callback[1] += 1
                    
                    if not image_ok:
                        dataset_complete_for_callback[0] = False

                pool_over_imgs(subdataset.image_dir,
                               check_image,
                               f'Checking images from {dataset.model_category} sub-dataset {subdataset_i}',
                               args,
                               callback,
                               dataset_complete)


                if dataset_complete[0] and dataset_complete[1] >= (args.resolution ** 3):
                    subdataset.rendering_complete = True
                    subdataset.save()
                else:
                    print('Not all images rendered')
                    continue
            
            if args.render_check:
                continue

            if subdataset.rendering_complete and not subdataset.images_compressed:

                pool_over_imgs(subdataset.image_dir,
                               compress_image,
                               f'Compressing images from {dataset.model_category} sub-dataset {subdataset_i}',
                               args)

                subdataset.images_compressed = True
                subdataset.save()

            if subdataset.images_compressed and not subdataset.images_grouped:

                img_dim = 224 - (2 * args.crop)

                subdataset.image_group.arr = np.empty((subdataset.resolution ** 3, img_dim, img_dim), dtype=np.uint8)

                def callback(args):
                    img, i = args

                    subdataset.image_group.arr[i] = img

                pool_over_imgs(subdataset.image_dir,
                               npImage,
                               f'Grouping images from {dataset.model_category} sub-dataset {subdataset_i}',
                               args,
                               callback)

                subdataset.image_group.dump()
                subdataset.image_group.arr = None
                subdataset.images_grouped = True
                subdataset.save()
