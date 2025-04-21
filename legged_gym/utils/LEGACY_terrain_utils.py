# def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
#     """
#     Generate a uniform noise terrain

#     Parameters
#         terrain (SubTerrain): the terrain
#         min_height (float): the minimum height of the terrain [meters]
#         max_height (float): the maximum height of the terrain [meters]
#         step (float): minimum height change between two points [meters]
#         downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

#     """
#     if downsampled_scale is None:
#         downsampled_scale = terrain.horizontal_scale

#     # switch parameters to discrete units
#     min_height = int(min_height / terrain.vertical_scale)
#     max_height = int(max_height / terrain.vertical_scale)
#     step = int(step / terrain.vertical_scale)

#     heights_range = np.arange(min_height, max_height + step, step)
#     height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
#         terrain.length * terrain.horizontal_scale / downsampled_scale)))

#     x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
#     y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

#     f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

#     x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
#     y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
#     z_upsampled = np.rint(f(y_upsampled, x_upsampled))

#     terrain.height_field_raw += z_upsampled.astype(np.int16)
#     return terrain


# def sloped_terrain(terrain, slope=1):
#     """
#     Generate a sloped terrain

#     Parameters:
#         terrain (SubTerrain): the terrain
#         slope (int): positive or negative slope
#     Returns:
#         terrain (SubTerrain): update terrain
#     """

#     x = np.arange(0, terrain.width)
#     y = np.arange(0, terrain.length)
#     xx, yy = np.meshgrid(x, y, sparse=True)
#     xx = xx.reshape(terrain.width, 1)
#     max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
#     terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
#     return terrain

# def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
#     """
#     Generate a sloped terrain

#     Parameters:
#         terrain (terrain): the terrain
#         slope (int): positive or negative slope
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     x = np.arange(0, terrain.width)
#     y = np.arange(0, terrain.length)
#     center_x = int(terrain.width / 2)
#     center_y = int(terrain.length / 2)
#     xx, yy = np.meshgrid(x, y, sparse=True)
#     xx = (center_x - np.abs(center_x-xx)) / center_x
#     yy = (center_y - np.abs(center_y-yy)) / center_y
#     xx = xx.reshape(terrain.width, 1)
#     yy = yy.reshape(1, terrain.length)
#     max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
#     terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

#     platform_size = int(platform_size / terrain.horizontal_scale / 2)
#     x1 = terrain.width // 2 - platform_size
#     x2 = terrain.width // 2 + platform_size
#     y1 = terrain.length // 2 - platform_size
#     y2 = terrain.length // 2 + platform_size

#     min_h = min(terrain.height_field_raw[x1, y1], 0)
#     max_h = max(terrain.height_field_raw[x1, y1], 0)
#     terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
#     return terrain

# def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
#     """
#     Generate a terrain with gaps

#     Parameters:
#         terrain (terrain): the terrain
#         max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
#         min_size (float): minimum size of a rectangle obstacle [meters]
#         max_size (float): maximum size of a rectangle obstacle [meters]
#         num_rects (int): number of randomly generated obstacles
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     # switch parameters to discrete units
#     max_height = int(max_height / terrain.vertical_scale)
#     min_size = int(min_size / terrain.horizontal_scale)
#     max_size = int(max_size / terrain.horizontal_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     (i, j) = terrain.height_field_raw.shape
#     height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
#     width_range = range(min_size, max_size, 4)
#     length_range = range(min_size, max_size, 4)

#     for _ in range(num_rects):
#         width = np.random.choice(width_range)
#         length = np.random.choice(length_range)
#         start_i = np.random.choice(range(0, i-width, 4))
#         start_j = np.random.choice(range(0, j-length, 4))
#         terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

#     x1 = (terrain.width - platform_size) // 2
#     x2 = (terrain.width + platform_size) // 2
#     y1 = (terrain.length - platform_size) // 2
#     y2 = (terrain.length + platform_size) // 2
#     terrain.height_field_raw[x1:x2, y1:y2] = 0
#     return terrain

# def wave_terrain(terrain, num_waves=1, amplitude=1.):
#     """
#     Generate a wavy terrain

#     Parameters:
#         terrain (terrain): the terrain
#         num_waves (int): number of sine waves across the terrain length
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     amplitude = int(0.5*amplitude / terrain.vertical_scale)
#     if num_waves > 0:
#         div = terrain.length / (num_waves * np.pi * 2)
#         x = np.arange(0, terrain.width)
#         y = np.arange(0, terrain.length)
#         xx, yy = np.meshgrid(x, y, sparse=True)
#         xx = xx.reshape(terrain.width, 1)
#         yy = yy.reshape(1, terrain.length)
#         terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
#             terrain.height_field_raw.dtype)
#     return terrain

# def stairs_terrain(terrain, step_width, step_height):
#     """
#     Generate a stairs

#     Parameters:
#         terrain (terrain): the terrain
#         step_width (float):  the width of the step [meters]
#         step_height (float):  the height of the step [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     # switch parameters to discrete units
#     step_width = int(step_width / terrain.horizontal_scale)
#     step_height = int(step_height / terrain.vertical_scale)

#     num_steps = terrain.width // step_width
#     height = step_height
#     for i in range(num_steps):
#         terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
#         height += step_height
#     return terrain

# def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
#     """
#     Generate stairs

#     Parameters:
#         terrain (terrain): the terrain
#         step_width (float):  the width of the step [meters]
#         step_height (float): the step_height [meters]
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     # switch parameters to discrete units
#     step_width = int(step_width / terrain.horizontal_scale)
#     step_height = int(step_height / terrain.vertical_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     height = 0
#     start_x = 0
#     stop_x = terrain.width
#     start_y = 0
#     stop_y = terrain.length
#     while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
#         start_x += step_width
#         stop_x -= step_width
#         start_y += step_width
#         stop_y -= step_width
#         height += step_height
#         terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
#     return terrain

# def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
#     """
#     Generate a stepping stones terrain

#     Parameters:
#         terrain (terrain): the terrain
#         stone_size (float): horizontal size of the stepping stones [meters]
#         stone_distance (float): distance between stones (i.e size of the holes) [meters]
#         max_height (float): maximum height of the stones (positive and negative) [meters]
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#         depth (float): depth of the holes (default=-10.) [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     # switch parameters to discrete units
#     stone_size = int(stone_size / terrain.horizontal_scale)
#     stone_distance = int(stone_distance / terrain.horizontal_scale)
#     max_height = int(max_height / terrain.vertical_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)
#     height_range = np.arange(-max_height-1, max_height, step=1)

#     start_x = 0
#     start_y = 0
#     terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
#     if terrain.length >= terrain.width:
#         while start_y < terrain.length:
#             stop_y = min(terrain.length, start_y + stone_size)
#             start_x = np.random.randint(0, stone_size)
#             # fill first hole
#             stop_x = max(0, start_x - stone_distance)
#             terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
#             # fill row
#             while start_x < terrain.width:
#                 stop_x = min(terrain.width, start_x + stone_size)
#                 terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
#                 start_x += stone_size + stone_distance
#             start_y += stone_size + stone_distance
#     elif terrain.width > terrain.length:
#         while start_x < terrain.width:
#             stop_x = min(terrain.width, start_x + stone_size)
#             start_y = np.random.randint(0, stone_size)
#             # fill first hole
#             stop_y = max(0, start_y - stone_distance)
#             terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
#             # fill column
#             while start_y < terrain.length:
#                 stop_y = min(terrain.length, start_y + stone_size)
#                 terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
#                 start_y += stone_size + stone_distance
#             start_x += stone_size + stone_distance

#     x1 = (terrain.width - platform_size) // 2
#     x2 = (terrain.width + platform_size) // 2
#     y1 = (terrain.length - platform_size) // 2
#     y2 = (terrain.length + platform_size) // 2
#     terrain.height_field_raw[x1:x2, y1:y2] = 0
#     return terrain