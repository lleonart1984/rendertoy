import math

from tutorials.lesson_common import ROOT_DIR

import rendering as ren
import time
import numpy as np


from PIL import Image
image_for_texture1 = np.array(Image.open(f"{ROOT_DIR}/models/marble1.jpg"))
image_for_texture2 = np.array(Image.open(f"{ROOT_DIR}/models/marble2.jpg"))
image_for_texture3 = np.array(Image.open(f"{ROOT_DIR}/models/marble3.jpg"))


@ren.kernel_main
def make_egg(vertices: [ren.MeshVertex], a: np.float32, b: np.float32):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    if(u <= b) p = (float3)(u, -sqrt(a * a - u * u * a * a / (b * b)), 0);
    else p = p = (float3)(2 * b - u, sqrt(a * a - (2 * b - u) * (2 * b - u) * a * a / (b * b)), 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """

@ren.kernel_main
def make_conic1(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.2, b = 0.6;
    if(u <= a) p = (float3)(u, 0.2, 0);
    else if(u <= b) p = (float3)(u, -0.5 * u + 0.3, 0);
    else p = (float3)(0.6 - (u - 0.6) * 1.5, 0, 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """


@ren.kernel_main
def make_conic2(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.2, b = 0.6, c = 0.7;
    if(u <= a) p = (float3)(u, 0.2, 0);
    else if(u <= b)
    {
        float x = 0.2 + (u - a) * 0.25;
        float y = 4 * x - 0.6;
        p = (float3)(x, y, 0);
    }
    else if(u <= c) p = (float3)(0.3, 0.6 + (u - b), 0);
    else p = (float3)((1.0 - u), 0.7, 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """

@ren.kernel_main
def make_plate(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.3, b = 0.65, c = 0.7;
    if(u <= a) p = (float3)(u, 0.7, 0);
    else if(u <= b)
    {
        float x = 0.3 + (u - a) * 12.0 / 7.0;
        float y = x * x * 5 / 18.0 + 27.0 / 40.0;
        p = (float3)(x, y, 0);
    }
    else if(u <= c) p = (float3)(0.9 - (u - b) * 2.0, 0.9, 0);
    else
    {
        float x = 0.8 - (u - c) * 8.0 / 3.0;
        float y = x * x * 5 / 18.0 + 27.0 / 40.0 + 0.05;
        p = (float3)(x, y, 0);
    }

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """

egg1_mesh = ren.manifold(200, 200)
egg1 = egg1_mesh.vertices
make_egg[egg1.shape](egg1, np.float32(0.65), np.float32(0.5))

conic1_mesh = ren.manifold(300, 300)
conic1 = conic1_mesh.vertices
make_conic1[conic1.shape](conic1)

conic2_mesh = ren.manifold(300, 300)
conic2 = conic2_mesh.vertices
make_conic2[conic2.shape](conic2)

plate_mesh = ren.manifold(300, 300)
plate = plate_mesh.vertices
make_plate[plate.shape](plate)

texture_memory1, texture_descriptor1 = ren.create_texture2D(image_for_texture1.shape[1], image_for_texture1.shape[0])
with ren.mapped(texture_memory1) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture1.shape[0], image_for_texture1.shape[1], 4)
    map[:,:,0:3] = image_for_texture1 / 255.0   # update rgb from image
    map[:,:,3] = 1.0    # set alphas = 1.0

texture_memory2, texture_descriptor2 = ren.create_texture2D(image_for_texture2.shape[1], image_for_texture2.shape[0])
with ren.mapped(texture_memory2) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture2.shape[0], image_for_texture2.shape[1], 4)
    map[:,:,0:3] = image_for_texture2 / 255.0   # update rgb from image
    map[:,:,3] = 1.0    # set alphas = 1.0

texture_memory3, texture_descriptor3 = ren.create_texture2D(image_for_texture3.shape[1], image_for_texture3.shape[0])
with ren.mapped(texture_memory3) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture3.shape[0], image_for_texture3.shape[1], 4)
    map[:,:,0:3] = image_for_texture3 / 255.0   # update rgb from image
    map[:,:,3] = 1.0    # set alphas = 1.0

texture_memory4, texture_descriptor4 = ren.create_texture2D(image_for_texture3.shape[1], image_for_texture3.shape[0])
with ren.mapped(texture_memory4) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture3.shape[0], image_for_texture3.shape[1], 4)
    map[:,:,0:3] = image_for_texture3 / 255.0   # update rgb from image
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

    with ren.mapped(vertex_shader_globals) as map:
        map["World"] = ren.matmul(
            ren.scale(1/3), ren.rotate(t, ren.make_float3(0, 1, 0))
        )
        map["View"] = ren.look_at(
            ren.make_float3(0, 0.5, 1.0),
            ren.make_float3(0, 0.2, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    ren.clear(raster.get_render_target())
    ren.clear(raster.get_depth_buffer(), 1.0)

    '''
    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor1.get()
    #raster.draw_points(conic1)
    raster.draw_triangles(conic1, conic1_mesh.indices)

    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor2.get()
    #raster.draw_points(conic2)
    raster.draw_triangles(conic2, conic2_mesh.indices)

    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor3.get()
    #raster.draw_points(plate)
    raster.draw_triangles(plate, plate_mesh.indices)
    '''


    with ren.mapped(vertex_shader_globals) as map:
        map["World"] = ren.matmul(ren.scale(1/3), ren.rotate(t, ren.make_float3(0, 1, 0)))

    with ren.mapped(fragment_shader_globals) as map:
        map["DiffuseMap"] = texture_descriptor4.get()
    raster.draw_triangles(egg1, egg1_mesh.indices)


    # Using a rasterizer to draw the point instead of handling everything by ourself.

    #raster.draw_triangles(vertex_buffer, index_buffer)

    presenter.present()




print("[INFO] Terminated...")