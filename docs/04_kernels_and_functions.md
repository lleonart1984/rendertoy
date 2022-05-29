# Inside kernels and functions in rendertoy

The main goal of `rendertoy` is to support an efficient (and fast) execution of drawing
algorithms. Parallel programming is crucial to this end and therefore, a new way of thinking
in logics is mandatory. To have an idea, the sum of all values of an array is really difficult
to think outside a simple accumulator logic, but this sequential logic is not efficient in parallel
programming. Instead, a reduction is used, summing independent pairs in parallel and repeat the 
operation until one single value remains.

In next sections we will discuss more in details some computer-graphics related algorithms and
how the parallelism helps to accelerate the overall process. Nevertheless, in this section we
will discuss the capabilities of `rendertoy` to express kernels and helper functions.

## The Mandelbrot example

A common example of parallelism is the synthesis of a fractal image. Normally these images can
be generated independently one pixel from another. A complex sequence in the form:

$$
Z_n = Z_{n-1}^2 + C
$$

generates a fractal when considered the convergence of the sequence. There are two images can be
generated, the one where each pixel is a value of C 
(assuming the pixel position x,y as the Cartesian representation of complex C) and initial 
$Z_0=(0,0)$, or assume the pixel corresponds to the initial value $Z_0$ and C a constant
value (that's the case in our example).

First, we will define a fancy function to map a color to a complex modulus.

```python
@ren.kernel_function
def get_color(m: np.float32) -> ren.float4:
    """
    m = min(m, 10.0f);
    float s = 2*(1.0f / (1 + exp(-m)) - 0.5f);
    return (float4)(0.0f, 1.0f-s, fmod(s+0.5,1.0), 1.0f);
    """
```

Notice, a kernel function can (and normally must) return a value. On the other hand, different
from kernels, those functions can not be dispatched from host application, they are only used to 
generate the backing code and inspect the name and signature if necessary. Thus, calling a 
kernel function directly from the host is always an error.

## Calling kernel functions

To call a kernel function from a kernel main is as usual by name. The only aspect to consider
here is that helper functions can not receive image objects. Also, the order used by a script 
to declare (decorate) functions that will produce the GPU program is the same of those functions
in the final code, therefore, the helper functions must be declared before used.

Saying this we can define our fractal image generator. First, we will define a struct used
to pass as a single parameter all the settings of the algorithm. This is useful to easily
change/add parameters of the algorithm without changing the main signature, neither adding 
a lot of arguments.

```python
# Define a struct to define parameters in the algorithm.
@ren.kernel_struct
class MandelbrotInfo:
    N: int
    C: ren.float2
```

Declaring an OpenCL struct from python can be done by our `rendertoy` decorator, `kernel_struct`.
Here, all fields annotations are used to declare the struct fields. Notice this struct has no any
logic valid for python (they are class fields instead of instance fields). Nevertheless, behind the
decorated type a numpy dtype object is kept, representing the type of the struct.

That means that we can use `MandelbrotInfo` as dtype of numpy array and the elements of the array
will hold a field `N` of type `np.int32` and a field `C` of type `dtype([('x','<f4),('y','<f4')])`.

Also, we can use it to create buffers or structs in `rendertoy`.

```python
mandelbrot_info = ren.create_struct(MandelbrotInfo)
```
Now, the content of `mandelbrot_info` is on the GPU (or device memory in general) and needs to be 
mapped in order to update its values.

```python
with ren.mapped(mandelbrot_info) as map:
    map["N"] = 100
    map["C"] = ren.make_float2(0.09, 0.61)
```

Notice that numpy accesses to the fields by indexing, while OpenCL accesses to the fields by
dot notation.

Finally, the kernel main function might look like next.

```python
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
```

## Rendering to a window!

An interesting animation comes from rendering different fractals for different values of C.
In order to have an interactive visualization we require from a window. In `rendertoy` project
we work with SDL windows API that allows windows creation cross-platform.

The required initializations are encapsulated in a concept `Presenter`. A presenter is created
with some specification on the image size, and then exposes an image as render target.
After the image is rendered, a `present` call will transfer the content to the window for 
visualization. Normally, a swapchain can be used to allow to work in the synthesis of an image
while another image of the swapchain is being presented. In our project this is not supported.

In our case, the presenter has also the role of an event manager for simplicity. The next code
shows the common use of the presenter in an animation loop.

```python
presenter = ren.create_presenter(512, 512)

while True:
    # poll events in the event queue of the window.
    event, arg = presenter.poll_events()
    # the unique event handled now is window closed
    if event == ren.Event.CLOSED:
        break

    render_target = presenter.get_render_target()
    
    # Draw something to render_target image.
    
    presenter.present()
```

Because we want to render the fractal in the render target, the code to substitute the drawing
can be:

```python
compute_mandelbrot[presenter.get_render_target().shape](
        render_target, mandelbrot_info
    )
```

If `mandelbrot_info` is updated every frame (specially the field C) based on some elapsed time
an animation is visualized.

The code here is exposed in the tutorial/lesson04_mandelbrot_animation.py script.

