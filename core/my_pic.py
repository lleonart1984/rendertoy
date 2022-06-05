from tutorials.lesson_common import ROOT_DIR

import rendering as ren
import time
import numpy as np


# Create a manifold mesh to represent the surface.
egg_mesh = ren.manifold(100, 100)
egg = egg_mesh.vertices

@ren.kernel_main
def make_egg(vertices: [ren.MeshVertex], a: np.float32, b: np.float32):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    if(u <= b) p = (float3)(u, -sqrt(a * a - u * u * a * a / (b * b)), 0);
    else p = p = (float3)(b - u, sqrt(a * a - (b - u) * (b - u) * a * a / (b * b)), 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 position = h.xyz;

    vertices[thread_id].P = position; // update position of the mesh with computed parametric transformation
    """

conic1_mesh = ren.manifold(100, 100)
conic1 = conic1_mesh.vertices

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

    float3 position = h.xyz;

    vertices[thread_id].P = position; // update position of the mesh with computed parametric transformation
    """

conic2_mesh = ren.manifold(100, 100)
conic2 = conic2_mesh.vertices

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

    float3 position = h.xyz;

    vertices[thread_id].P = position; // update position of the mesh with computed parametric transformation
    """

plate_mesh = ren.manifold(100, 100)
plate = plate_mesh.vertices

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

    float3 position = h.xyz;

    vertices[thread_id].P = position; // update position of the mesh with computed parametric transformation
    """


@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


# Create the buffer to store the matrices
transform_info = ren.create_struct(Transforms)


@ren.kernel_main
def transform_and_draw(
    im: ren.w_image2d_t, vertices: [ren.MeshVertex], info: Transforms
):
    """
    int2 dim = get_image_dim(im);
    float3 P = vertices[thread_id].P;
    float3 C = (float3)(1.0, 1.0, 0.3);

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info.World); // transform with respect to world matrix
    H = mul(H, info.View);  // transform with respect to view matrix
    H = mul(H, info.Proj);  // transform with respect to projection matrix

    H.xyz /= H.w; // De-homogenize

    if (any(H.xyz < (float3)(-1.0, -1.0, 0.0)) || any(H.xyz >= 1))
    return;  // clip if outside normalized device coordinate box

    int px = (int)(dim.x * (H.x * 0.5 + 0.5));
    int py = (int)(dim.y * (0.5 - H.y * 0.5));

    write_imagef(im, (int2)(px,py), (float4)(C.x, C.y, C.z, 1.0));
    """


# creates a window with a 512x512 image as render target.
presenter = ren.create_presenter(640, 480)

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

    #make_egg[egg.shape](egg, np.float32(0.65), np.float32(0.5))
    make_conic1[conic1.shape](conic1)
    make_conic2[conic2.shape](conic2)
    make_plate[plate.shape](plate)

    # update the transformation matrices from host every frame
    with ren.mapped(transform_info) as map:
        map["World"] = ren.rotate(t, ren.make_float3(1, 0, 0))
        map["View"] = ren.look_at(
            ren.make_float3(0, 1, 5),
            ren.make_float3(0, 0, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    ren.clear(presenter.get_render_target())

    # transform_and_draw[egg.shape](presenter.get_render_target(), egg, transform_info)
    transform_and_draw[conic1.shape](presenter.get_render_target(), conic1, transform_info)
    transform_and_draw[conic2.shape](presenter.get_render_target(), conic2, transform_info)
    transform_and_draw[plate.shape](presenter.get_render_target(), plate, transform_info)

    presenter.present()

print("[INFO] Terminated...")