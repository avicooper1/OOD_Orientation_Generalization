import bpy
import os
import json
import sys
import pandas as pd
import numpy as np

dir = os.path.dirname(bpy.data.filepath)
print("1111")
print(dir)
print("2222")
if not dir in sys.path:
    sys.path.append(dir)

sys.path.append('/home/avic/Rotation-Generalization')
sys.path.append('/home/avic/Rotation-Generalization/render')
sys.path.append('/home/avic/om2/datasets')
print('g')
from generate_sm_object import new_stimulus, stim_ids
print('h')
from dataset_path import DatasetPath, DatasetType
print('i')

REDUCE_OUTPUT = True
MODEL_ZOO_PATHS = '/home/avic/Rotation-Generalization/render/model_paths2.json'
SHAPENET_PATH = '/home/avic/om5/ShapeNetCore.v2/'

if not os.path.exists(MODEL_ZOO_PATHS):
    print("Error: no model zoo path file exists")
    exit(1)

with open(MODEL_ZOO_PATHS, 'r') as m:
    model_paths = json.load(m)

JOB_ID = int(sys.argv[-3])
MODEL_I = JOB_ID % 50

#EXPLANATION Setting correct paths

DD = DatasetPath.get_dataset(JOB_ID // 50)

BIN_RANGE = ((-0.25, 0.25) if not DD.type == DatasetType.Hole else (-1.8, -1.3), (-0.25, 0.25))

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
print('e')
print(sys.argv[-2])
image_data = pd.read_csv(sys.argv[-2]).iloc[int(sys.argv[-1])]
print('f')
obj.rotation_euler = [image_data.object_x, image_data.object_y, image_data.object_z]
if DD.scale:
    if 'object_scale' in image_data.index:
        obj.scale = [image_data.object_scale for _ in range(3)]
    else:
        obj_scale = 0.825 if DD.model_category != 'SM' else 0.04125

print('a')
# if REDUCE_OUTPUT:
#     # redirect output to log file
#     logfile = 'blender_render.log'
#     open(logfile, 'a').close()
#     old = os.dup(1)
#     sys.stdout.flush()
#     os.close(1)
#     os.open(logfile, os.O_WRONLY)

print('b')

# do the rendering
bpy.context.scene.render.filepath = image_data.image_name
print(image_data.image_name)
bpy.ops.render.render(write_still=True)
# exit(0)

print('c')

# if REDUCE_OUTPUT:
#     # disable output redirection
#     os.close(1)
#     os.dup(old)
#     os.close(old)

print("asdasd")