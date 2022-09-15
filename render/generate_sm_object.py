import bpy
import numpy as np
import random

# Attribution: Code taken from https://github.com/jhamrick/shepard-metzler-stimuli

def gen_stim_id():
    stim_id = ""
    last_direction = None
    last_step = None
    last_was_block = True
    directions = ['x','y','z']
    for x in range(0, random.randint(3,7)):
        direction = random.choice(directions)

        if 0 == random.randint(0,3) and not last_was_block:
            step = random.choice([-1, 1])
            stim_id += f'{direction}_{step}_{last_direction}_{-last_step}_'
            last_was_block = True
        else:
            step = random.choice([y for y in range(-4, 5) if y < -1 or y > 1])
            last_step = step
            stim_id += f'{direction}_{step}_'
            if last_direction:
                directions.append(last_direction)
            last_direction = directions.pop(directions.index(direction))
            last_was_block = False
    return stim_id[:-1]

stim_ids = [
     'z_2_y_-3_x_-4_z_-1_x_4_z_4',
     'y_-4_z_3_y_-1_z_-3_x_4_y_-4',
     'y_3_x_1_y_-3_z_-4_y_-2_x_-3_y_2_x_-1_y_-2',
     'y_3_z_1_y_-3_z_2_y_3_z_-3_x_2',
     'y_3_x_-4_y_1_x_4_y_-3_x_-4',
     'x_2_z_-2_y_2_x_-3_y_-3',
     'z_-2_y_1_z_2_y_-2_z_-4_y_3',
     'y_2_z_-3_y_-2',
     'z_4_x_-2_y_4',
     'y_-4_x_-4_z_-3_y_-2',
     'z_2_x_-2_z_-3',
     'y_-2_z_2_x_-2_z_2_y_1_z_-2_y_4_z_2',
     'y_-2_z_3_x_1_z_-3',
     'x_3_y_3_x_-3_z_-2',
     'x_2_z_2_y_-4_z_3_x_-4_y_4',
     'y_4_z_-3_x_-3_y_3_z_3_y_-1_z_-3_x_-4',
     'x_4_y_1_x_-4_z_-3',
     'x_-2_z_-3_y_3_x_3_z_-1_x_-3',
     'z_2_x_-1_z_-2_y_2',
     'z_2_y_4_x_-3_y_-3',
     'y_-2_x_-3_z_-3_x_-4_y_2_x_3_y_-1_x_-3',
     'x_4_z_1_x_-4_z_-4_x_3_y_3_z_4_x_-3',
     'y_-2_z_1_y_2_x_4_y_1_x_-4_y_2_x_-2',
     'x_-4_z_-4_y_-1_z_4_y_2_z_1_y_-2_x_-2_z_-4',
     'x_2_y_-4_z_-1_y_4_z_-4_y_2_z_-4',
     'x_3_z_-2_x_4',
     'y_2_x_2_z_-4_y_2',
     'x_4_z_-3_x_-3_y_-3_x_2_z_-1_x_-2_y_-4',
     'z_-3_x_4_z_3_x_3_z_3',
     'x_-3_z_3_x_3_z_-4_x_-2_z_3',
     'x_3_z_1_x_-3_z_2_x_-2_y_-2_z_-4',
     'z_-3_y_2_x_3_y_1_x_-3_z_2',
     'z_2_y_4_x_1_y_-4',
     'y_4_z_3_x_-4_y_4_x_-1_y_-4',
     'z_-4_y_1_z_4_x_-4_y_3',
     'z_3_y_2_z_1_y_-2_x_3_z_2_y_-2_x_2',
     'y_-2_z_1_y_2_x_-3_y_2_x_2_z_-1_x_-2_z_-3',
     'x_-3_z_3_y_4_z_3',
     'y_-2_x_-3_y_-3_x_-1_y_3_x_3_y_4_z_-3',
     'x_2_z_3_y_4_x_-2',
     'y_3_z_4_y_-3_x_2',
     'y_3_z_4_y_-4',
     'z_-2_x_3_z_4_y_-2',
     'x_-3_z_4_y_-3_z_-4_x_-4',
     'z_2_y_-2_z_2',
     'y_2_x_-4_y_-2_z_1_y_2_z_4',
     'y_2_x_-1_y_-2_z_4_x_4',
     'y_3_z_-2_y_-1_z_2_y_3_z_-1_y_-3_z_-2',
     'y_-2_x_-3_y_-4_z_4_x_3_z_-1_x_-3_y_4',
     'z_-4_x_-1_z_4_x_-3_y_1_x_3_y_4_z_-1_y_-4']

def parse_stim_id(stim_id):
    parts = stim_id.split("_")
    if parts[0] == "A":
        reflect = False
    elif parts[0] == "B":
        reflect = True
    else:
        raise ValueError(stim_id)
    parts = parts[1:]
    block_locs = [(0, 0, 0)]

    for i in range(0, len(parts), 2):
        coord = parts[i]
        num = int(parts[i + 1])
        if num < 0:
            direction = -1
        else:
            direction = 1
        scale = 2 * direction
        for _ in range(np.abs(num)):
            x, y, z = block_locs[-1]
            if coord == 'x':
                x += scale
            elif coord == 'y':
                y += scale
            elif coord == 'z':
                z += scale
            block_locs.append((x, y, z))

    # we actually have one block too many, so get rid of the first block and
    # then adjust the rest of the blocks
    block_locs = np.array(block_locs[1:], dtype=float)
    block_locs -= block_locs[0]

    # now center the block locations
    center = (block_locs.min(axis=0) + block_locs.max(axis=0)) / 2.0
    block_locs -= center

    # reflect across the y-axis
    if reflect:
        block_locs[:, 1] *= -1

    return block_locs


def new_stimulus(stim_id):
    scene = bpy.data.scenes["Scene"]

    cube = bpy.data.objects["Cube"]

    # parse the stim id into block locations
    block_locs = parse_stim_id(stim_id)
    cube.location = block_locs[0]

    # create copies of the cube to form the full stimulus
    blocks = [cube]
    for i in range(1, len(block_locs)):
        name = "block_{:02d}".format(i)
        mesh = bpy.data.meshes.new(name)
        ob_new = bpy.data.objects.new(name, mesh)
        ob_new.data = cube.data.copy()
        ob_new.data.name = "{}_data".format(name)
        ob_new.scale = cube.scale
        ob_new.location = block_locs[i]
        scene.collection.objects.link(ob_new)
        blocks.append(ob_new)

    # join the blocks together
    bpy.ops.object.select_all(action='DESELECT')
    for block in blocks:
        block.select_set(state=True)
    bpy.context.view_layer.objects.active = blocks[-1]
    bpy.ops.object.join()

    # remove double vertices
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.editmode_toggle()

    return bpy.context.view_layer.objects.active