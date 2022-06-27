import math
from tutorials.lesson_common import ROOT_DIR
from perlin_noise_textures import marble, wood
from figures_modelation import *
from utils import *
import rendering as ren
import time
import numpy as np
from PIL import Image

vertex_shader_globals = ren.create_struct(Transforms)
fragment_shader_globals = ren.create_struct(Materials)

########################################
###           Textures               ###
########################################
images_folder = f"{ROOT_DIR}/Class2022/Claudia Olavarrieta - Marcos Valdivie/images"
image_for_texture1 = np.array(Image.open(f"{images_folder}/marble1.jpg"))
image_for_texture2 = np.array(Image.open(f"{images_folder}/marble2.jpg"))
image_for_texture3 = np.array(Image.open(f"{images_folder}/marble3.jpg"))
image_for_texture4 = marble(200, 200, 0.0, 1.0, 1.5, 32.0)
image_for_texture5 = 200 - marble(200, 200, 1.0, 1.0, 7.0, 32.0) * np.array([121, 50, 21]) * 200 / (256 * 256)
image_for_texture6 = 200 - (256 - marble(200, 200, 0.0, 2.0, 9.0, 32.0)) * np.array([135, 206, 235]) * 200 / (256 * 256)
image_for_texture7 = np.array(Image.open(f"{images_folder}/wood.jpg"))
image_for_texture8 = np.repeat([135, 135, 120], 40000).reshape((200, 200, 3)).astype(np.uint8)
texture_memory1, texture_descriptor1 = create_and_map_textures(image_for_texture1)
texture_memory2, texture_descriptor2 = create_and_map_textures(image_for_texture2)
texture_memory3, texture_descriptor3 = create_and_map_textures(image_for_texture3)
texture_memory4, texture_descriptor4 = create_and_map_textures(image_for_texture4)
texture_memory5, texture_descriptor5 = create_and_map_textures(image_for_texture5)
texture_memory6, texture_descriptor6 = create_and_map_textures(image_for_texture6)
texture_memory7, texture_descriptor7 = create_and_map_textures(image_for_texture7)
texture_memory8, texture_descriptor8 = create_and_map_textures(image_for_texture8)

########################################
###           Meshes                 ###
########################################
conic1_mesh = ren.manifold(200, 200)
conic1 = conic1_mesh.vertices
make_conic1[conic1.shape](conic1)

conic2_mesh = ren.manifold(200, 200)
conic2 = conic2_mesh.vertices
make_conic2[conic2.shape](conic2)

plate_mesh = ren.manifold(200, 200)
plate = plate_mesh.vertices
make_plate[plate.shape](plate)

table_mesh = ren.manifold(225, 225)
table = table_mesh.vertices

back_mesh = ren.manifold(200, 200)
back = back_mesh.vertices

egg1_mesh = ren.manifold(200, 200)
egg1 = egg1_mesh.vertices
make_egg[egg1.shape](egg1)

egg1_mesh = ren.manifold(200, 200)
egg1 = egg1_mesh.vertices
make_egg[egg1.shape](egg1)

egg1_mesh = ren.manifold(200, 200)
egg1 = egg1_mesh.vertices
make_egg[egg1.shape](egg1)

egg2_mesh = ren.manifold(200, 200)
egg2 = egg2_mesh.vertices
make_egg[egg2.shape](egg2)

egg3_mesh = ren.manifold(200, 200)
egg3 = egg3_mesh.vertices
make_egg[egg3.shape](egg3)


########################################
###        Object Positioning        ###
########################################
transformations = ren.to_array(ren.scale(3/5)) @ \
                ren.to_array(ren.rotate(math.pi, ren.make_float3(0, 1, 0))) @ \
                ren.to_array(ren.rotate(2 / 5 * math.pi, ren.make_float3(0, 0, 1))) @ \
                ren.to_array(ren.translate(ren.make_float3(-0.4, 1.05, 0)))
transform[egg1.shape](egg1, transformations)

transformations = ren.to_array(transformations) @ \
                  ren.to_array(ren.rotate(2 / 3 * math.pi, ren.make_float3(0, 1, 0)))
transform[egg2.shape](egg2, transformations)

transformations = ren.to_array(transformations) @ \
                  ren.to_array(ren.rotate(2 / 3 * math.pi, ren.make_float3(0, 1, 0)))
transform[egg3.shape](egg3, transformations)

transformations = ren.to_array(ren.translate(ren.make_float3(-0.5, -0.5, 0))) @ \
                  ren.to_array(ren.rotate(math.pi / 2, ren.make_float3(1, 0, 0)))
transform[table.shape](table, transformations)

transformations = ren.to_array(ren.translate(ren.make_float3(-0.5, -0.5, 0))) @ \
                  ren.to_array(ren.scale(5)) @ \
                  ren.to_array(ren.translate(ren.make_float3(0, 0, -2)))
transform[back.shape](back, transformations)

transformations = ren.to_array(ren.scale(1/5))
transform[conic1.shape](conic1, transformations)
transform[conic2.shape](conic2, transformations)
transform[plate.shape](plate, transformations)
transform[egg1.shape](egg1, transformations)
transform[egg2.shape](egg2, transformations)
transform[egg3.shape](egg3, transformations)


with ren.mapped(table) as map:
    map['N'] = ren.normalize(ren.make_float3(0, 1, 0))


presenter = ren.create_presenter(640, 480)

raster = ren.Raster(
    presenter.get_render_target(),  # render target to draw on
    transform_and_draw,             # vertex shader used, only transform and set normal as a color
    vertex_shader_globals,          # buffer with the transforms
    fragment_to_color,              # fragment shader, only return the color of the vertex
    fragment_shader_globals         # buffer with the material texture
)


with ren.mapped(vertex_shader_globals) as map:
    map["World"] = ren.scale(1 / 2)
    map["View"] = ren.look_at(
        ren.make_float3(0, 0.25, 0.45),
        ren.make_float3(0, 0.1, 0),
        ren.make_float3(0, 1, 0),
    )
    map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

ren.clear(raster.get_render_target())
ren.clear(raster.get_depth_buffer(), 1.0)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor1.get()
raster.draw_triangles(conic1, conic1_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor2.get()
raster.draw_triangles(conic2, conic2_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor3.get()
raster.draw_triangles(plate, plate_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor4.get()
raster.draw_triangles(egg1, egg1_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor6.get()
raster.draw_triangles(egg2, egg2_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor5.get()
raster.draw_triangles(egg3, egg3_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor7.get()
raster.draw_triangles(table, table_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor8.get()
raster.draw_triangles(back, back_mesh.indices)

# save_cl_image(presenter.get_render_target(), 'final_image')

'''
raster.draw_points(conic1)
raster.draw_points(conic2)
raster.draw_points(plate)
raster.draw_points(egg1)
raster.draw_points(egg2)
raster.draw_points(egg3)
raster.draw_points(table)
raster.draw_points(back)
save_cl_image(presenter.get_render_target(), 'meshes')

marble = marble(640, 480, 0.0, 2.0, 3.0, 32.0)
save_image(marble, 'marble')
'''

presenter.present()

while True:
    event, arg = presenter.poll_events()
    if event == ren.Event.CLOSED:
        break