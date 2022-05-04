import lesson_common


import rendering as ren
import time
import numpy as np


"""
In this lesson, a point cloud is rendered after transformation
"""


# Define a struct to define parameters in the algorithm.
@ren.kernel_struct
class Vertex:
    P: ren.float3  # Position
    C: ren.float3  # Color


# Create vertex buffer
num_vertex = 10000
vertex_buffer = ren.create_buffer(num_vertex, Vertex)

# fill vertices with a sphere.
with ren.mapped(vertex_buffer) as map:
    # the number of float numbers in positions and colors are not 3 floats per vertex but 4 because of the padding imposse to float3
    map["P"] = (
        np.random.uniform(-1.0, 1.0, size=(num_vertex * 4,))
        .astype(np.float32)
        .view(ren.float3)
    )
    map["C"] = (
        np.random.uniform(0, 1, size=(num_vertex * 4,))
        .astype(np.float32)
        .view(ren.float3)
    )


@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


# Create the buffer to store the matrices
transform_info = ren.create_struct(Transforms)


@ren.kernel_main
def transform_and_draw(im: ren.w_image2d_t, vertices: [Vertex], info: Transforms):
    """
    int2 dim = get_image_dim(im);
    float3 P = vertices[thread_id].P;
    float3 C = vertices[thread_id].C;

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
presenter = ren.create_presenter(512, 512)

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
    with ren.mapped(transform_info) as map:
        map["World"] = ren.scale(0.2 + np.cos(t * 4) * 0.2, 0.2 + np.sin(t) * 0.1, 0.2)
        map["View"] = ren.translate(0, 0, 0.3)
        map["Proj"] = ren.identity()

    ren.clear(presenter.get_render_target())

    transform_and_draw[vertex_buffer.shape](
        presenter.get_render_target(), vertex_buffer, transform_info
    )

    presenter.present()
