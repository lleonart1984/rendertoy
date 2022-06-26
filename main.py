import time

import rendering as ren
import numpy as np
from PIL import Image


@ren.kernel_struct
class Vertex_Out:
    proj: ren.float4
    L: ren.float3
    C: ren.float2


@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


@ren.kernel_struct
class Materials:
    DiffuseMap: ren.Texture2D

# @ren.kernel_struct
# def casteljau(points, t):
# # https://en.wikipedia.org/wiki/Casteljau%27s_algorithm
#     n = len(points) - 1
#     if n == 0:
#         return points[0]
#     else:
#         return (1 - t) * casteljau(points[:n], t) + t * casteljau(points[1:], t)


@ren.kernel_main
def perform_parametric_transform(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float3 p = (float3)(0,0,0);
    float3 n = (float3)(0,0,0);
    float3 b = (float3)(0,0,0);
    float3 t = (float3)(0,0,0);

    // dispatch scene
    if(thread_id % 6 == 0 )
    {
        float u = uv.x*2*3.1416;
        float v = uv.y*5;
        p = (float3)(4*cos(u),4*sin(u),15+3+v);
        n = (float3)(cos(u), sin(u), 0);
        b = (float3)(1,0,0);
        t = (float3)(0,0,1);

    }
    if(thread_id % 6 == 1)
    {
        float u = uv.x*2*3.1416;
        float v = uv.y*5;
        p = (float3)(3.7*cos(u),3.7*sin(u),15+3+v);
        n = (float3)(-cos(u),  - sin(u), 0);
        b = (float3)(1,0,0);
        t = (float3)(0,0,1);

    }

    if(thread_id % 6 == 2)
    {
        float v = uv.y*4-2;
        float u =uv.x*3.1416*2;
        p  = (float3)(v*cos(u)*3.5,v*sin(u)*3.5,-v*v+25);
        n = (float3)(v*cos(u) + v*sin(u), 0, 0);
       b = (float3)(1,0,0);
        t = (float3)(0,0,1);

    }


      if(thread_id % 6 == 3)
    {
        float u = uv.x*3.1416*2;
        float v = uv.y*0.3-4;
        p = (float3)(v*cos(u),v*sin(u),15+3);
        n = (float3)(cos(u), sin(u), 0);
        b = (float3)(0,0,1);


    }

    if(thread_id % 6 == 4)
    {
        float v = uv.y*4-2;
        float u =uv.x*3.1416*2;
        p  = (float3)(v*cos(u)*3.5,v*sin(u)*3.5,-v*v+25.3);
        n = (float3)(v*cos(u) + v*sin(u), 0, 0);
        b = (float3)(1,0,0);
        t = (float3)(0,0,1);
    }

    if(thread_id % 6 == 5)
    {
        float u = uv.x*2*3.1416;
        float v = uv.y*0.3;
        p = (float3)(7*cos(u),7*sin(u),21+v);
        n = (float3)(cos(u), sin(u), 0);
        b = (float3)(1,0,0);
        t = (float3)(0,0,1);

    }

    vertices[thread_id].N = n;
    vertices[thread_id].T = t;
    vertices[thread_id].B = b;
    vertices[thread_id].P = p; // update position of the mesh with computed parametric transformation
    """


@ren.kernel_function  # Vertex shaders will be treated as kernel functions, the main function is implemented in the rasterizer
def transform_and_draw(vertex: ren.MeshVertex, info: Transforms) -> Vertex_Out:
    """
    float3 P = vertex.P;
    float d = 0.2f + max(0.0f, dot(vertex.N, normalize((float3)(1,1,1))));
    float3 L = (float3)(d, d, d); // vertex.N * 0.5f + 0.5f; // use normals as a color for debugging purposes

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info.World); // transform with respect to world matrix
    H = mul(H, info.View);  // transform with respect to view matrix
    H = mul(H, info.Proj);  // transform with respect to projection matrix

    Vertex_Out o;
    o.proj = H;
    o.L = L;
    if (vertex.P.z > 18.1 && (vertex.P.x*vertex.P.x + vertex.P.y*vertex.P.y < 48.9)) {
        o.C =  vertex.P.xz/4;
        }
    else
    {
        o.C = (float2)(0,0);
    }
 
    return o;
    """

@ren.kernel_function
def fragment_to_color(fragment: Vertex_Out, info: Materials) -> ren.float4:
    """
    float3 diff = sample2D(info.DiffuseMap, fragment.C).xyz;
    return (float4)(diff * fragment.L, 1);
    """


# Create a manifold mesh to represent the surface.
mesh = ren.manifold(3000, 3000)
control_points = ren.create_buffer(5, ren.float3)
vertex_buffer, index_buffer = mesh.vertices,None

image = Image.open(f"./models/porcelana.jpg")
image = image.resize((500, 500),resample=Image.Resampling.BILINEAR)
image_for_texture = np.array(image)
texture_memory, texture_descriptor = ren.create_texture2D(image_for_texture.shape[1], image_for_texture.shape[0])

with ren.mapped(texture_memory) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture.shape[0], image_for_texture.shape[1], 4)
    map[:, :, 0:3] = image_for_texture / 255.0  # update rgb from image
    map[:, :, 3] = 1  # set alphas = 1.0

# Create the buffer to store the matrices for the vertex shader
vertex_shader_globals = ren.create_struct(Transforms)
# Create the buffer to store the matrices for the vertex shader
shader_globals = ren.create_struct(Transforms)
# Create the buffer to store the matrices
transform_info = ren.create_struct(Transforms)
fragment_shader_globals = ren.create_struct(Materials)

# creates a window with a 512x512 image as render target.
presenter = ren.create_presenter(640, 480)

raster = ren.Raster(
    presenter.get_render_target(),  # render target to draw on
    transform_and_draw,  # vertex shader used, only transform and set normal as a color
    vertex_shader_globals,  # buffer with the transforms
    fragment_to_color,  # fragment shader, only return the color of the vertex
    fragment_shader_globals  # buffer with the material texture
)


perform_parametric_transform[vertex_buffer.shape](vertex_buffer)



# ------------------------- Animation
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
        map["World"] = ren.matmul(ren.rotate(t, ren.make_float3(0, 0, 1)),
                                  ren.rotate(99.5, ren.make_float3(-1, 0, 0))
                                  )
        map["View"] = ren.translate(0, 17, 10)
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor.get()

    ren.clear(raster.get_render_target())
    ren.clear(raster.get_depth_buffer(), 1.0)
    # Using a rasterizer to draw the point instead of handling everything by ourself.
    raster.draw_points(vertex_buffer)
    # raster.draw_triangles(vertex_buffer, None)

    presenter.present()

#
# ------------------------- only one frame
#
# # update the transformation matrices from host every frame
# with ren.mapped(vertex_shader_globals) as map:
#     map["World"] = ren.matmul(ren.rotate(10, ren.make_float3(0, 0, 1)), ren.rotate(99.5, ren.make_float3(-1, 0, 0)))
#     map["View"] = ren.translate(0, 17, 10)
#     map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)
#
# with ren.mapped(fragment_shader_globals) as map:
#     map["DiffuseMap"] = texture_descriptor.get()
#
# ren.clear(raster.get_render_target())
# ren.clear(raster.get_depth_buffer(), 2.0)
# # Using a rasterizer to draw the point instead of handling everything by ourself.
# raster.draw_points(vertex_buffer)
# presenter.present()
# input("Press Enter to continue...")
#
# print("[INFO] Terminated...")

