import bpy
import os
from random import uniform
import csv
import json
import sys
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

sys.path.append('/home/avic/Rotation-Generalization')
sys.path.append('/home/avic/Rotation-Generalization/render')

from generate_sm_object import new_stimulus, stim_ids
from dataset_path import DatasetPath, DatasetType

REDUCE_OUTPUT = True
MODEL_ZOO_PATHS = '/home/avic/Rotation-Generalization/render/model_paths2.json'
SHAPENET_PATH = '/home/avic/om5/ShapeNetCore.v2/'

if not os.path.exists(MODEL_ZOO_PATHS):
    print("Error: no model zoo path file exists")
    exit(1)

with open(MODEL_ZOO_PATHS, 'r') as m:
    model_paths = json.load(m)

JOB_ID = int(sys.argv[-1])
MODEL_I = JOB_ID % 50

#EXPLANATION Setting correct paths

DD = DatasetPath.get_dataset(JOB_ID // 50)

BIN_RANGE = ((-0.25, 0.25) if not DD.type == DatasetType.Hole else (-1.8, -1.3), (-0.25, 0.25))

if not os.path.exists(DD.path):
    os.makedirs(DD.path, exist_ok=True)
    with open(DD.config_file, 'w') as f:
        f.write(f"""\
MODEL_CATEGORY: {DD.model_category}
TYPE: {DD.type}
SCALE: {DD.scale}
RESTRICTION_AXES: {DD.restriction_axes}
BIN_RANGE: {BIN_RANGE}""")

os.makedirs(DD.image_path, exist_ok=True)


#EXPLANATION Check for already generated images

NUM_IMAGES_TO_GENERATE = 5_000 if DD.type == DatasetType.Full else 4_000
IMAGES_OFFSET = 0

DATASET_ALREADY_EXISTS = os.path.exists(DD.annotation_path_i(MODEL_I))
print(DD.annotation_path_i(MODEL_I))
if DATASET_ALREADY_EXISTS:
    d = pd.read_csv(DD.annotation_path_i(MODEL_I))
#
    if len(d) >= NUM_IMAGES_TO_GENERATE:
        print("No more images necessary to generate")
        exit(0)
    IMAGES_OFFSET += len(d)


#EXPLANATION Set up scene

def remove_obj(obj):
    bpy.ops.object.delete({"selected_objects": [obj]})

bpy.context.scene.render.image_settings.color_mode = 'BW'
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
bpy.context.scene.view_layers['View Layer'].cycles.use_denoising = True

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 400

# Select the computing device.
prefs = bpy.context.preferences.addons['cycles'].preferences
devices = prefs.get_devices()
bpy.context.scene.cycles.device = 'GPU'
prefs.compute_device_type = 'CUDA'
devices[0][0].use = True
devices[1][0].use = True

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

if DD.model_category != 'SM':
    remove_obj(cube)
    camera.location[1] = -4.7
camera.rotation_euler = np.pi / 2, 0, 0
camera.data.lens = 100
bpy.context.scene.view_settings.look = 'High Contrast'

bpy.ops.mesh.primitive_plane_add(enter_editmode=False, size=3, location=(0, 0, -1))
plane = objects['Plane']
for poly in plane.data.polygons: poly.select = False
for edge in plane.data.edges: edge.select = False
for vertex in plane.data.vertices: vertex.select = False
plane.data.edges[3].select = True
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={"use_normal_flip":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 3), "orient_type":'NORMAL', "orient_matrix":((0, -1, 0), (1, 0, -0), (0, 0, 1)), "orient_matrix_type":'NORMAL', "constraint_axis":(False, False, True), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":True, "use_accurate":False})
plane.modifiers.new(type='BEVEL', name='Bevel')
plane.modifiers["Bevel"].segments = 10
bpy.ops.material.new()
dark_shade = bpy.data.materials[-1]
dark_shade.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0699782, 0.0699782, 0.0699782, 1)
plane.data.materials.append(dark_shade)

if DD.model_category == 'SM':
    MODEL_NAME = f'SM{MODEL_I}'
    obj = new_stimulus(f'A_{stim_ids[MODEL_I]}')
else:
    MODEL_NAME, MODEL_PATH = model_paths[DD.model_category][MODEL_I]
    bpy.ops.import_scene.obj(filepath=os.path.join(MODEL_PATH, 'models', 'model_normalized.obj'))
    obj = objects['model_normalized']
obj.data.materials.clear()
obj.rotation_mode = 'ZYX'

with open(DD.annotation_path_i(MODEL_I), 'a') as d:

    writer = csv.writer(d)
    if not DATASET_ALREADY_EXISTS:
        header = ['image_name',
        'model_name',
        'object_x',
        'object_y',
        'object_z']
        if DD.scale and DD.type == DatasetType.Full: header.append('object_scale')
        writer.writerow(header)

    for i in range(IMAGES_OFFSET, NUM_IMAGES_TO_GENERATE):
        # time.sleep(0.05)
        obj_pose = R.random().as_euler('zyx')
        if not DD.type == DatasetType.Full:
            while not (BIN_RANGE[0][0] <= obj_pose[DD.restriction_axes[0]] <= BIN_RANGE[0][1] and BIN_RANGE[1][0] <= obj_pose[
                    DD.restriction_axes[1]] <= BIN_RANGE[1][1]):
                obj_pose = R.random().as_euler('zyx')
        obj_pose = [round(a, 5) for a in obj_pose]
        obj.rotation_euler = obj_pose

        if DD.scale:
            if DD.type == DatasetType.Full:
                obj_scale = round(uniform(0.65, 1), 5) if DD.model_category != 'SM' else round(uniform(0.0325, 0.05), 6)
            else:
                obj_scale = 0.825 if DD.model_category != 'SM' else 0.04125
            obj.scale = [obj_scale for _ in range(3)]

        image_name = os.path.join(DD.image_path, f'{DD.model_category}_{MODEL_I}_{i}.png')
        entries = [
            image_name,
            MODEL_NAME,
            obj_pose[0],
            obj_pose[1],
            obj_pose[2]
        ]

        if DD.scale and DD.type == DatasetType.Full:
            entries.append(obj_scale)

        writer.writerow(entries)
        bpy.context.scene.render.filepath = image_name

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

        if i % 100 == 0:
            print("a")
            d.flush()
            print("b")
            # typically the above line would do. however this is used to ensure that the file is written
            os.fsync(d.fileno())
            print("c")
