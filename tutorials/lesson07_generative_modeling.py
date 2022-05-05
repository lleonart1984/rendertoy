from lesson_common import ROOT_DIR

import rendering as ren
import time
import numpy as np


"""
In this lesson, a point cloud created with generative modeling techniques is shown. 
"""


# Create a manifold mesh to represent the surface.
mesh = ren.manifold(48, 48)

control_points = ren.create_buffer(5, ren.float3)


@ren.kernel_function
def C_n_k(n: int, k: int) -> int:
    """
    if (k < n - k) k = n - k;
    long f = 1;
    for (int i = k + 1; i <= n; i++)
        f *= i;
    for (int i = 2; i <= n - k; i++)
        f /= i;
    return (int)f;
    """


@ren.kernel_main
def perform_parametric_transform(vertices: [ren.MeshVertex], cps: [ren.float3], cp_count: int):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    // Evaluate bezier
    float t = u;
    float3 p = (float3)(0,0,0);
    int n = cp_count - 1;
    for (int k = 0; k <= n; k++)
        p += cps[k] * C_n_k(n, k) * pow(t, (float)k) * pow(1 - t, (float)(n - k));

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 position = h.xyz;

    vertices[thread_id].P = position; // update position of the mesh with computed parametric transformation
    """


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

    # update control points wrt the time
    with ren.mapped(control_points) as map:
        map[:] = [
            ren.make_float3(0, 0, 0),
            ren.make_float3(1, 0, 0),
            ren.make_float3(1 + np.sin(t*9)*0.4, 1 + np.sin(t * 2) * 0.3, 0),
            ren.make_float3(1 + np.cos(t*7)*0.2, 1.5 + np.cos(t*3)*0.4, 0),
            ren.make_float3(0, 1 + np.sin(t*4)*0.4, 0)
        ]

    perform_parametric_transform[vertex_buffer.shape](vertex_buffer, control_points, len(control_points))

    # update the transformation matrices from host every frame
    with ren.mapped(transform_info) as map:
        map["World"] = ren.matmul(
            ren.scale(1.0), ren.rotate(t, ren.normalize(ren.make_float3(1, 1, 0)))
        )
        map["View"] = ren.look_at(
            ren.make_float3(0, 3.5, 4.5),
            ren.make_float3(0, 1, 0),
            ren.make_float3(0, 1, 0),
        )
        map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

    ren.clear(presenter.get_render_target())

    transform_and_draw[vertex_buffer.shape](
        presenter.get_render_target(), vertex_buffer, transform_info
    )

    presenter.present()

print("[INFO] Terminated...")