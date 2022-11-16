import bpy
import os
import csv
import json
import sys
import numpy as np
from tqdm import tqdm
import argparse

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render images for a given dataset')
    parser.add_argument('project_path', help='path to the project directory')
    parser.add_argument('storage_path', help='path to the storage directory')
    parser.add_argument('resolution', type=int, help='dataset resolution')
    parser.add_argument('dataset', type=str, help='dataset to render')
    parser.add_argument('sub_dataset', type=int, help='sub-dataset to render')

    args = parser.parse_args(sys.argv[5:])

    sys.path.append(args.project_path)

    from render.generate_sm_object import new_stimulus, stim_ids
    from utils.persistent_data_class import ImageDataset
    from utils.tools import get_heatmap_cell_ranges, range_mid

    REDUCE_OUTPUT = True
    MODEL_ZOO_PATHS = os.path.join(args.project_path, 'render/model_paths.json')
    SHAPENET_PATH = '/home/avic/om5/ShapeNetCore.v2/'

    if not os.path.exists(MODEL_ZOO_PATHS):
        print("Error: no model zoo path file exists")
        exit(1)

    with open(MODEL_ZOO_PATHS, 'r') as m:
        model_paths = json.load(m)

    IMAGE_DATASET = ImageDataset(args.storage_path, args.dataset, args.resolution)
    SUB_IMAGE_DATASET = IMAGE_DATASET.subdatasets[args.sub_dataset]

    # Set up scene
    def remove_obj(obj):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.view_layers['ViewLayer'].cycles.use_denoising = True
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 400

    bpy.context.scene.render.use_persistent_data = True

    scene = bpy.data.scenes['Scene']
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0

    objects = bpy.data.objects
    light = objects['Light']
    remove_obj(light)
    bpy.ops.object.light_add(type='POINT', radius=4, location=(5, -1, 0))
    objects[-1].data.energy = 1000
    bpy.ops.object.light_add(type='POINT', radius=4, location=(-5, -1, 1))
    objects[-1].data.energy = 1000
    bpy.ops.object.light_add(type='POINT', radius=4, location=(0, -3.5, 0))
    objects[-1].data.energy = 100
    camera = objects['Camera']
    cube = objects['Cube']

    camera.location = 0, -2.5, 0

    if IMAGE_DATASET.model_category != 'SM':
        remove_obj(cube)
    camera.rotation_euler = np.pi / 2, 0, 0
    camera.data.lens = 100
    bpy.context.scene.view_settings.look = 'High Contrast'

    if IMAGE_DATASET.model_category == 'SM':
        MODEL_NAME = f'SM{args.sub_dataset}'
        obj = new_stimulus(f'A_{stim_ids[args.sub_dataset]}')
    else:
        MODEL_NAME, MODEL_PATH = model_paths[IMAGE_DATASET.model_category][args.sub_dataset]
        bpy.ops.import_scene.obj(filepath=os.path.join(MODEL_PATH, 'models', 'model_normalized.obj'))
        obj = objects['model_normalized']

    obj.data.materials.clear()
    obj.rotation_mode = 'ZYX'
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = 0, 0, 0
    DIAGONAL_TARGET_LENGTH = 0.75 if IMAGE_DATASET.model_category != 'SM' else 0.7
    dimension_scale_factor = np.sqrt((DIAGONAL_TARGET_LENGTH ** 2) / np.sum(np.array(obj.dimensions) ** 2))
    obj.dimensions = [obj.dimensions[i] * dimension_scale_factor for i in range(3)]

    cubelet_ranges = get_heatmap_cell_ranges(args.resolution).reshape(args.resolution ** 3, 3, 2)

    DATASET_ALREADY_EXISTS = SUB_IMAGE_DATASET.annotation_file.on_disk()

    with open(SUB_IMAGE_DATASET.annotation_file.file_path, 'a') as d:

        writer = csv.writer(d, delimiter=';')

        if not DATASET_ALREADY_EXISTS:
            writer.writerow(('image_name', 'instance_name', 'object_x', 'object_y', 'object_z', 'cubelet_i'))

        for cubelet_i, cubelet in tqdm(enumerate(cubelet_ranges), total=cubelet_ranges.shape[0]):

            image_file_name = SUB_IMAGE_DATASET.image_path(cubelet_i)
            image_path = os.path.join(args.storage_path, image_file_name)

            if os.path.exists(image_path) and os.stat(image_path).st_size > 0:
                continue

            obj_pose = [round(range_mid(r), 5) for r in cubelet]
            obj.rotation_euler = obj_pose

            entries = [
                image_file_name,
                MODEL_NAME,
                obj_pose[0],
                obj_pose[1],
                obj_pose[2],
                np.unravel_index(cubelet_i, (args.resolution, args.resolution, args.resolution))
            ]

            writer.writerow(entries)
            bpy.context.scene.render.filepath = image_path

            if REDUCE_OUTPUT:
                # redirect output to log file
                logfile = 'blender_render.log'
                open(logfile, 'a').close()
                old = os.dup(1)
                sys.stdout.flush()
                os.close(1)
                os.open(logfile, os.O_WRONLY)

            # do the rendering
            bpy.ops.render.render(write_still=True)

            if REDUCE_OUTPUT:
                # disable output redirection
                os.close(1)
                os.dup(old)
                os.close(old)

            if cubelet_i % 100 == 0:
                d.flush()
