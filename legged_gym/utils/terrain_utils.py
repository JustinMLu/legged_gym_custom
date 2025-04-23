# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None):
    """
    Generate a uniform noise terrain.
    
    Parameters:
        terrain (SubTerrain): the terrain object
        min_height (float): minimum terrain height [meters]
        max_height (float): maximum terrain height [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points
                                   (must be ≥ terrain.horizontal_scale)
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # Switch parameters to discrete units.
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)
    heights_range = np.arange(min_height, max_height + step, step)

    # Compute downsampled grid dimensions.
    down_rows = int(terrain.length * terrain.horizontal_scale / downsampled_scale)
    down_cols = int(terrain.width * terrain.horizontal_scale / downsampled_scale)
    height_field_downsampled = np.random.choice(heights_range, (down_rows, down_cols))

    # Create coordinate arrays.
    # y-axis corresponds to terrain.length (rows) and x-axis to terrain.width (cols)
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, down_rows)
    x = np.linspace(0, terrain.width * terrain.horizontal_scale, down_cols)

    # Create the interpolation function.
    # interp2d expects (x, y, z) where z has shape (len(y), len(x))
    f = interpolate.interp2d(x, y, height_field_downsampled, kind='linear')

    # Upsample coordinates to the full resolution of the SubTerrain.
    up_x = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    up_y = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    # f(up_x, up_y) returns an array of shape (len(up_y), len(up_x))
    z_upsampled = np.rint(f(up_x, up_y))

    # Add the upsized patch to the terrain height field.
    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain

def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    length, width = terrain.height_field_raw.shape
    max_h = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * width)
    # gradient along columns (width)
    grad = (np.arange(width) / (width - 1)) * max_h
    terrain.height_field_raw += grad.astype(terrain.height_field_raw.dtype)[None, :]
    return terrain

def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    """Pyramid slope rising toward centre then flattened on a square/rect platform."""
    length, width = terrain.height_field_raw.shape
    ctr_x = width  // 2
    ctr_y = length // 2

    # Normalised distances [0..1] to centre along each axis
    x = (ctr_x - np.abs(np.arange(width)  - ctr_x)) / ctr_x
    y = (ctr_y - np.abs(np.arange(length) - ctr_y)) / ctr_y
    yy, xx = np.meshgrid(y, x, indexing="ij")  # shape (length,width)

    max_h = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (width / 2))
    terrain.height_field_raw += (max_h * xx * yy).astype(terrain.height_field_raw.dtype)

    # clip a flat platform in the centre
    half = int(platform_size / terrain.horizontal_scale / 2)
    x1, x2 = ctr_x - half, ctr_x + half
    y1, y2 = ctr_y - half, ctr_y + half
    min_h = min(terrain.height_field_raw[y1, x1], 0)
    max_h = max(terrain.height_field_raw[y1, x1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain

def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.0):
    """Random rectangles of ±height scattered across the map."""
    # convert → cells
    h_max = int(max_height / terrain.vertical_scale)
    min_s = int(min_size   / terrain.horizontal_scale)
    max_s = int(max_size   / terrain.horizontal_scale)
    plat   = int(platform_size / terrain.horizontal_scale)

    length, width = terrain.height_field_raw.shape
    height_choices = [-h_max, -h_max // 2, h_max // 2, h_max]

    for _ in range(num_rects):
        w = np.random.choice(range(min_s, max_s, 4))  # rectangle width  (cols)
        l = np.random.choice(range(min_s, max_s, 4))  # rectangle length (rows)
        row0 = np.random.choice(range(0, length - l, 4))
        col0 = np.random.choice(range(0, width  - w, 4))
        terrain.height_field_raw[row0:row0 + l, col0:col0 + w] = np.random.choice(height_choices)

    # keep central platform clear
    cx1 = (width  - plat) // 2
    cx2 = (width  + plat) // 2
    cy1 = (length - plat) // 2
    cy2 = (length + plat) // 2
    terrain.height_field_raw[cy1:cy2, cx1:cx2] = 0
    return terrain

def wave_terrain(terrain, num_waves=1, amplitude=1.0):
    """Sinusoidal surface on both axes."""
    amp = int(0.5 * amplitude / terrain.vertical_scale)
    if num_waves <= 0:
        return terrain

    length, width = terrain.height_field_raw.shape
    div_y = length / (num_waves * 2 * np.pi)
    div_x = width  / (num_waves * 2 * np.pi)

    yy, xx = np.meshgrid(np.arange(length), np.arange(width), indexing="ij")
    surf = amp * (np.cos(yy / div_y) + np.sin(xx / div_x))
    terrain.height_field_raw += surf.astype(terrain.height_field_raw.dtype)
    return terrain

def stairs_terrain(terrain, step_width, step_height):
    """Stairs that rise along *length* (forward) direction."""
    step_w = int(step_width  / terrain.horizontal_scale)   # depth (rows) per step
    step_h = int(step_height / terrain.vertical_scale)

    length, _ = terrain.height_field_raw.shape
    num_steps = length // step_w
    height = step_h
    for s in range(num_steps):
        r0 = s * step_w
        r1 = (s + 1) * step_w
        terrain.height_field_raw[r0:r1, :] += height
        height += step_h
    return terrain

def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """A pyramidal staircase stepping up toward the centre then back down."""
    step_w = int(step_width  / terrain.horizontal_scale)
    step_h = int(step_height / terrain.vertical_scale)
    plat   = int(platform_size / terrain.horizontal_scale)

    length, width = terrain.height_field_raw.shape
    top = 0
    r0, r1 = 0, length
    c0, c1 = 0, width
    while (r1 - r0) > plat and (c1 - c0) > plat:
        r0 += step_w; r1 -= step_w
        c0 += step_w; c1 -= step_w
        top += step_h
        terrain.height_field_raw[r0:r1, c0:c1] = top
    return terrain

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    stone_sz  = int(stone_size      / terrain.horizontal_scale)
    stone_gap = int(stone_distance  / terrain.horizontal_scale)
    h_max     = int(max_height      / terrain.vertical_scale)
    plat      = int(platform_size   / terrain.horizontal_scale)
    pit_depth = int(depth           / terrain.vertical_scale)

    length, width = terrain.height_field_raw.shape
    terrain.height_field_raw[:] = pit_depth

    height_choices = np.arange(-h_max - 1, h_max, 1)
    row = 0
    while row < length:
        row_end = min(length, row + stone_sz)
        col = np.random.randint(0, stone_sz)
        # first gap in row
        gap_end = max(0, col - stone_gap)
        terrain.height_field_raw[row:row_end, 0:gap_end] = np.random.choice(height_choices)
        while col < width:
            col_end = min(width, col + stone_sz)
            terrain.height_field_raw[row:row_end, col:col_end] = np.random.choice(height_choices)
            col += stone_sz + stone_gap
        row += stone_sz + stone_gap

    # central safe platform
    cx1 = (width  - plat) // 2
    cx2 = (width  + plat) // 2
    cy1 = (length - plat) // 2
    cy2 = (length + plat) // 2
    terrain.height_field_raw[cy1:cy2, cx1:cx2] = 0
    return terrain

# ================================= EXTREME PARKOUR =================================
def parkour_hurdle_terrain_randomized(terrain,
                           platform_len=2.5,
                           platform_height=0.5,
                           x_range=(14.0, 14.1),
                           y_range=(-6.0, -5.9), # as centered as I could make it
                           num_hurdles=1,
                           hurdle_thickness=0.3,
                           hurdle_height_range=(0.2, 0.3),
                           half_valid_width=(2.4, 2.5),
                           border_width=0.1,
                           border_height=0.5):
    """ Parkour hurdle course with vertical hurdles.

        The robot starts on a raised *platform*, then confronts `num_hurdles`
        evenly spaced hurdle blocks.  A lateral corridor (gap) of constant
        width is cut through each hurdle so the robot has a path.

        Args are in **meters**; converted to grid cells using the
        SubTerrain class's scales.
    """
    # Buffer to store hurdle XY positions
    terrain.hurdles = []

    # Midline of the y-axis
    mid_y = terrain.width // 2

    # World units → grid cell conversions
    h_scale = terrain.horizontal_scale # [m] → cols/rows
    v_scale = terrain.vertical_scale   # [m] → height

    x_min = round(x_range[0] / h_scale)
    x_max = round(x_range[1] / h_scale)
    y_min = round(y_range[0] / h_scale)
    y_max = round(y_range[1] / h_scale)

    # Random valid half-width for the gap in each hurdle
    half_gap = round(np.random.uniform(half_valid_width[0],
                                       half_valid_width[1]) / h_scale)

    # Hurdle height bounds in grid units
    hurdle_h_min = round(hurdle_height_range[0] / v_scale)
    hurdle_h_max = round(hurdle_height_range[1] / v_scale)

    # Build the initial starting platform
    platform_cells = round(platform_len / h_scale)
    platform_h = round(platform_height / v_scale)
    terrain.height_field_raw[:platform_cells, :] = platform_h

    # Stone width in cells
    stone_cells = round(hurdle_thickness / h_scale)

    # Start at the end of platform
    current_x = platform_cells

    # Place each hurdle
    for i in range(num_hurdles):

        # Randomly select (x,y) in range
        dx = np.random.randint(x_min, x_max)
        dy = np.random.randint(y_min, y_max)
        current_x += dx

        
        # Raise the hurdle
        h_choice = np.random.randint(hurdle_h_min, hurdle_h_max)
        x_start = current_x - stone_cells // 2
        x_end = current_x + stone_cells // 2

        # Carve out the gap around mid_y + dy
        terrain.height_field_raw[x_start:x_end, :] = h_choice
        terrain.height_field_raw[
            x_start:x_end, :mid_y + dy - half_gap
        ] = 0
        terrain.height_field_raw[
            x_start:x_end, mid_y + dy + half_gap:
        ] = 0

        # Convert grid coordinates to world coordinates
        x_local = current_x * terrain.horizontal_scale
        y_local = (mid_y + dy) * terrain.horizontal_scale

        # Add hurdle positions to terrain object
        terrain.hurdles.append((x_local, y_local))

        # Print
        print(f"Hurdle {i+1} placed at env coordinates: ({x_local:.2f}, {y_local:.2f})")


    # Final platform at end
    final_dx = np.random.randint(x_min, x_max)
    final_x = current_x + final_dx
    max_x = terrain.width - round(0.5 / h_scale)
    final_x = min(final_x, max_x)

    # Add padding walls around the perimeter
    pad_cells = int(border_width / h_scale)
    pad_h = int(border_height / v_scale)

    hf = terrain.height_field_raw
    hf[:, :pad_cells] = pad_h        # left
    hf[:, -pad_cells:] = pad_h       # right
    hf[:pad_cells, :] = pad_h        # bottom
    hf[-pad_cells:, :] = pad_h       # top

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.5,
                           x_positions=[7.0, 11.0, 14.5],  # EXACT X positions for each hurdle
                           y_positions=[0.0, 0.0, 0.0],     # EXACT Y positions for each hurdle
                           hurdle_thickness=0.5,
                           hurdle_heights=None,  # NEW: list of hurdle heights in meters for each hurdle
                           half_valid_width=2.5,
                           border_width=0.1,
                           border_height=0.5):
    """Parkour terrain with hurdles at exact specified positions.
    
    Args:
        x_positions: List of exact X coordinates for each hurdle [meters]
        y_positions: List of exact Y coordinates for each hurdle [meters]
        hurdle_heights: Optional list of exact hurdle heights (in meters) for each hurdle.
                        If not provided, a default hurdle_height is used for all hurdles.
        (Other args in meters, converted to grid cells)
    """
    # Validate inputs
    num_hurdles = len(x_positions)
    assert len(y_positions) == num_hurdles, "x_positions and y_positions must have the same length"
    if hurdle_heights is not None:
        assert len(hurdle_heights) == num_hurdles, "hurdle_heights must have same length as x_positions"

    # Buffer to store hurdle positions
    terrain.hurdle_positions = []

    # Midline of the y-axis (terrain.width)
    mid_y = terrain.width // 2

    # World units → grid cell conversions
    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale

    # Build the initial starting platform
    platform_cells = round(platform_len / h_scale)
    platform_h = round(platform_height / v_scale)
    terrain.height_field_raw[:platform_cells, :] = platform_h

    # Stone width in cells (for the hurdle thickness)
    stone_cells = round(hurdle_thickness / h_scale)

    # Fixed half-width for the gap in hurdles
    half_gap = round(half_valid_width / h_scale)

    # Place each hurdle at the EXACT specified positions
    for i in range(num_hurdles):
        # Convert world X,Y to grid cells
        x_local = x_positions[i]
        y_local = y_positions[i]
        
        current_x = round(x_local / h_scale)
        current_y = mid_y + round(y_local / h_scale)
        
        # Determine hurdle height in grid units:
        hurdle_h_i = round(hurdle_heights[i] / v_scale)
        
        # Define x-interval for the hurdle
        x_start = current_x - stone_cells // 2
        x_end = current_x + stone_cells // 2

        # "Raise" the hurdle: set the hurdle area to the hurdle height
        terrain.height_field_raw[x_start:x_end, :] = hurdle_h_i

        # Carve out the gap so the robot has a corridor: clear the area around current_y
        terrain.height_field_raw[x_start:x_end, :current_y - half_gap] = 0
        terrain.height_field_raw[x_start:x_end, current_y + half_gap:] = 0

        # Store the exact world coordinates for this hurdle
        terrain.hurdle_positions.append((x_local, y_local))

    # Add padding walls around the perimeter
    pad_cells = int(border_width / h_scale)
    pad_h = int(border_height / v_scale)
    hf = terrain.height_field_raw
    hf[:, :pad_cells] = pad_h        # left
    hf[:, -pad_cells:] = pad_h       # right
    hf[:pad_cells, :] = pad_h        # bottom
    hf[-pad_cells:, :] = pad_h       # top

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles

# ================================ SUB-TERRAIN CLASS ================================
class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        # Swap dimensions: use (length, width) so height_field_raw rows=length and cols=width
        self.height_field_raw = np.zeros((self.length, self.width), dtype=np.int16)
# ===================================================================================
