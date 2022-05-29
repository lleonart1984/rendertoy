from lesson_common import ROOT_DIR

import rendering as ren
import time
import numpy as np


"""
In this lesson, a point cloud loaded from a wavefront obj file is rasterized with two shaders as points or triangles
"""


# Load vertex buffer from obj
visuals = ren.load_obj(f"{ROOT_DIR}/models/dragon.obj")
mesh, material = visuals[0]
vertex_buffer, index_buffer = mesh.vertices, None


@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


@ren.kernel_struct
class Vertex_Out:
    proj: ren.float4
    C: ren.float3


# Create the buffer to store the matrices for the vertex shader
shader_globals = ren.create_struct(Transforms)


@ren.kernel_function  # Vertex shaders will be treated as kernel functions, the main function is implemented in the rasterizer
def transform_and_draw(
    vertex: ren.MeshVertex, info: Transforms
) -> Vertex_Out:
    """
    float3 P = vertex.P;
    float d = max(0.2f, dot(vertex.N, normalize((float3)(1,1,1))));
    float3 C = (float3)(d,d,d); // vertex.N * 0.5f + 0.5f; // use normals as a color for debugging purposes

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info.World); // transform with respect to world matrix
    H = mul(H, info.View);  // transform with respect to view matrix
    H = mul(H, info.Proj);  // transform with respect to projection matrix

    Vertex_Out o;
    o.proj = H;
    o.C = C;
    return o;
    """


@ren.kernel_function
def fragment_to_color(fragment: Vertex_Out, info: Transforms) -> ren.float4:
    """
    return (float4)(fragment.C.x, fragment.C.y, fragment.C.z, 1);
    """


# creates a window with a 512x512 image as render target.
presenter = ren.create_presenter(640, 480)

raster = ren.Raster(
    presenter.get_render_target(),  # render target to draw on
    transform_and_draw,             # vertex shader used, only transform and set normal as a color
    shader_globals,          # buffer with the transforms
    fragment_to_color,              # fragment shader, only return the color of the vertex
    shader_globals                            # unused buffer
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
    with ren.mapped(shader_globals) as map:
        map["World"] = ren.matmul(
            ren.scale(1.0), ren.rotate(t, ren.make_float3(0, 1, 0))
        )
        map["View"] = ren.look_at(
            ren.make_float3(0, 0.3, 1.0),
            ren.make_float3(0, 0, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    ren.clear(raster.get_render_target())
    ren.clear(raster.get_depth_buffer(), 1.0)
    # Using a rasterizer to draw the point instead of handling everything by ourself.
    # raster.draw_points(vertex_buffer)
    raster.draw_triangles(vertex_buffer, index_buffer)

    presenter.present()
