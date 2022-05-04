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
The logic behind a presenter is to update the render target every frame and present the image in a window.
Normally APIs provide for mechanisms to have several images in a chain and generate and present images in parallel.
"""


# Define a struct to define parameters in the algorithm.
@ren.kernel_struct
class MandelbrotInfo:
    N: int
    C: ren.float2


mandelbrot_info = ren.create_struct(MandelbrotInfo)


@ren.kernel_function
def get_color(m: np.float32) -> ren.float4:
    """
    m = min(m, 10.0f);
    float s = 2*(1.0f / (1 + exp(-m)) - 0.5f);
    return (float4)(0.0f, 1.0f-s, fmod(s+0.5,1.0), 1.0f);
    """


@ren.kernel_main
def compute_mandelbrot(im: ren.w_image2d_t, info: MandelbrotInfo):
    """
    int2 dim = get_image_dim(im);
    // in rendering, only linear layout of threads is allowed. Mapping to image positions needs to be done manually.
    int px = thread_id % dim.x;
    int py = thread_id / dim.x;

    float2 Z = ((float2)((px + 0.5f)/dim.x, (py + 0.5f)/dim.y)) * 2.0f - 1.0f;

    for (int i=0; i<info.N; i++)
        Z = (float2)(Z.x*Z.x - Z.y*Z.y, 2*Z.x*Z.y) + info.C;

    float m = sqrt(dot(Z, Z));

    write_imagef(im, (int2)(px,py), get_color(m));
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

    with ren.mapped(mandelbrot_info) as map:
        map["N"] = 100
        map["C"] = np.array(
            (0.09 + np.cos(t * 0.7) * 0.01, 0.61 + np.sin(t) * 0.01), dtype=ren.float2
        )

    compute_mandelbrot[presenter.get_render_target().shape](
        presenter.get_render_target(), mandelbrot_info
    )

    presenter.present()
