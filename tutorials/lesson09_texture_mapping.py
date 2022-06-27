from lesson_common import ROOT_DIR

import rendering as ren
import time
import numpy as np


"""
In this lesson, a point cloud loaded from a wavefront obj file is rasterized with two shaders as a point
"""


# Load vertex buffer from obj
visuals = ren.load_obj(f"{ROOT_DIR}/models/dragon.obj")
mesh, material = visuals[0]
vertex_buffer, index_buffer = mesh.vertices, None


## Creating texture from scratch
# Textures in render toy are always float4 images. Notice the difference with respect to images in opencl.
# Because images in OpenCL can not be used in non-kernel functions we will use a memory pool accessible from anywhere
# in the code and a texture descriptor to know how to locate a texture (offset, width and height).
# For simplicity during sampling we will fix the supported format to only 4 floats r,g,b,a.
# # Next call creates a 4x4 (float4) texture and returns the memory object
# # (opencl buffer with pixels and a descriptor, opencl buffer with the footprint of the texture within the memory pool)
# texture_memory, texture_descriptor = ren.create_texture2D(4, 4)
# # Map the memory object to access the pixels as a numpy array
# with ren.mapped(texture_memory) as map:
#     map[:,:] = ren.make_float4(1,1,0,1) # Write yellow to all texels
#     map[0,0] = ren.make_float4(1,0,0,1) # Write red to texel at 0,0

# Next call creates a wxh (float4) texture and returns the memory object
from PIL import Image
image_for_texture = np.array(Image.open(f"{ROOT_DIR}/models/marble2.jpg"))
texture_memory, texture_descriptor = ren.create_texture2D(image_for_texture.shape[1], image_for_texture.shape[0])
with ren.mapped(texture_memory) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture.shape[0], image_for_texture.shape[1], 4)
    map[:,:,0:3] = image_for_texture/255.0   # update rgb from image
    map[:,:,3] = 1.0    # set alphas = 1.0


@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


@ren.kernel_struct
class Materials:
    DiffuseMap: ren.Texture2D


@ren.kernel_struct
class Vertex_Out:
    proj: ren.float4
    L: ren.float3
    C: ren.float2


# Create the buffer to store the matrices for the vertex shader
vertex_shader_globals = ren.create_struct(Transforms)
fragment_shader_globals = ren.create_struct(Materials)


@ren.kernel_function  # Vertex shaders will be treated as kernel functions, the main function is implemented in the rasterizer
def transform_and_draw(
    vertex: ren.MeshVertex, info: Transforms
) -> Vertex_Out:
    """
    float3 P = vertex.P;
    float d = 0.2f + max(0.0f, dot(vertex.N, normalize((float3)(1,1,1))));
    float3 L = (float3)(d,d,d); // vertex.N * 0.5f + 0.5f; // use normals as a color for debugging purposes

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info.World); // transform with respect to world matrix
    H = mul(H, info.View);  // transform with respect to view matrix
    H = mul(H, info.Proj);  // transform with respect to projection matrix

    Vertex_Out o;
    o.proj = H;
    o.L = L;
    o.C = vertex.P.xy * 2;
    return o;
    """


@ren.kernel_function
def fragment_to_color(fragment: Vertex_Out, info: Materials) -> ren.float4:
    """
    float3 diff = sample2D(info.DiffuseMap, fragment.C).xyz;
    return (float4)(diff * fragment.L, 1);
    """


# creates a window with a 512x512 image as render target.
presenter = ren.create_presenter(640, 480)

raster = ren.Raster(
    presenter.get_render_target(),  # render target to draw on
    transform_and_draw,             # vertex shader used, only transform and set normal as a color
    vertex_shader_globals,          # buffer with the transforms
    fragment_to_color,              # fragment shader, only return the color of the vertex
    fragment_shader_globals         # buffer with the material texture
)

# get the start time to compute the time for the animation
start_time = time.perf_counter()

while True:
    # poll events in the event queue of the window.
    event, arg = presenter.poll_events()
    # only event handled is window closed
    if event == ren.Event.CLOSED:
        break

    # t is the elapsed time
    t = time.perf_counter() - start_time

    # update the transformation matrices from host every frame
    with ren.mapped(vertex_shader_globals) as map:
        map["World"] = ren.matmul(
            ren.scale(1.0), ren.rotate(t, ren.make_float3(0, 1, 0))
        )
        map["View"] = ren.look_at(
            ren.make_float3(0, 0.3, 1.0),
            ren.make_float3(0, 0, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor.get()

    ren.clear(raster.get_render_target())
    ren.clear(raster.get_depth_buffer(), 1.0)
    # Using a rasterizer to draw the point instead of handling everything by ourself.
    # raster.draw_points(vertex_buffer)
    raster.draw_triangles(vertex_buffer, index_buffer)

    presenter.present()
