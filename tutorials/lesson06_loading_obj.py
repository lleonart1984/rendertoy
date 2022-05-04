ROOT_DIR = "./"  # Used to find the models folder

# If file is run as a script add parent directory to path
# This allow import rendering module
if __name__ == "__main__":
    import sys
    import os
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    ROOT_DIR = str(parentdir)

import rendering as ren
import time
import numpy as np


"""
In this lesson, a point cloud loaded from a wavefront obj file is rendered after transformation
"""


# Load vertex buffer from obj
visuals = ren.load_obj(f"{ROOT_DIR}/models/dragon.obj")
mesh, material = visuals[0]
vertex_buffer = mesh.vertices


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
    float3 C = vertices[thread_id].N * 0.5f + 0.5f; // use normals as a color for debugging purposes

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

    # update the transformation matrices from host every frame
    with ren.mapped(transform_info) as map:
        map["World"] = ren.matmul(
            ren.scale(1.0), ren.rotate(t, ren.make_float3(0, 1, 0))
        )
        map["View"] = ren.look_at(
            ren.make_float3(0, 0.3, 2),
            ren.make_float3(0, 0, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    ren.clear(presenter.get_render_target())

    transform_and_draw[vertex_buffer.shape](
        presenter.get_render_target(), vertex_buffer, transform_info
    )

    presenter.present()
