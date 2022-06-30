import numpy as np
from PIL import Image
import rendering as ren


def create_and_map_textures(image):
    texture_memory, texture_descriptor = ren.create_texture2D(image.shape[1], image.shape[0])
    with ren.mapped(texture_memory) as map:
        # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
        map = map.view(np.float32).ravel().reshape(image.shape[0], image.shape[1], 4)
        map[:, :, 0:3] = image / 255.0  # update rgb from image
        map[:, :, 3] = 1.0  # set alphas = 1.0
    return texture_memory, texture_descriptor


images_folder = "./images"

image_for_texture_alcohol = np.array(Image.open(f"{images_folder}/btears2.jpg"))
texture_memory_alcohol, texture_descriptor_alcohol = create_and_map_textures(image_for_texture_alcohol)
image_for_texture_dark_wood = np.array(Image.open(f"{images_folder}/dark_wood.webp"))
texture_memory_dark_wood, texture_descriptor_dark_wood = create_and_map_textures(image_for_texture_dark_wood)
image_for_texture_yellow = np.array(Image.open(f"{images_folder}/yellow.jpg"))
texture_memory_yellow, texture_descriptor_yellow = create_and_map_textures(image_for_texture_yellow)
image_for_texture_glass = np.array(Image.open(f"{images_folder}/glass_top2.jpg"))
texture_memory_glass, texture_descriptor_glass = create_and_map_textures(image_for_texture_glass)
image_for_texture_glass_down = np.array(Image.open(f"{images_folder}/glass_top2.jpg"))
texture_memory_glass_down, texture_descriptor_glass_down = create_and_map_textures(image_for_texture_glass_down)
image_for_texture_bottle = np.array(Image.open(f"{images_folder}/bottle.jpg"))
texture_memory_bottle, texture_descriptor_bottle = create_and_map_textures(image_for_texture_bottle)
image_for_texture_bottle_cap = np.array(Image.open(f"{images_folder}/glass_top.jpg"))
texture_memory_bottle_cap, texture_descriptor_bottle_cap = create_and_map_textures(image_for_texture_bottle_cap)
