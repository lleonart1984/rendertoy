# Introduction

The main goal of the `rendertoy` project is to support the understanding 
of elementary computer graphics concepts via re-implementation. Commonly used objects
in black-boxes APIs such as buffers, textures, descriptors, shaders, rasterization, 
tessellators, raycasters, are recreated in this library to be understood and usable
in a specific rendering task.

The project stands on top of cross-platform python libraries pyopencl and numpy.
Exploiting parallel programming through GPUs allows the rendering frame-rates 
to be competitive even with traditional low-level APIs.

## The two worlds

In parallel programming we always need to think in two computational models but also
in two different (physical) computing devices. Normal application logics lives in 
CPU accesible memory but highly-efficient parallel programming is only accessible
through GPU programming and therefore, objects and programs must be moved to the GPU.

The library `rendertoy` hides the use of OpenCL for programming kernel logics and 
CPU-GPU memory transfers but is helpful to have that in mind. In next, if a logic or
memory lives on the parallel-suitable device we will refer as "device", if the logic 
or memory lives of the application cpu-side we will refer as "host" (Similar naming
convention used in most of the parallel technologies).

## Memory

In `rendertoy` the creation of a device buffer is via the function `create_buffer`.

```python
import rendering as ren
import numpy as np
x = ren.create_buffer(100, np.float32)
```

In here, a buffer of 100s float values is created on the device (commonly GPU memory).
In order to populate the buffer with values, we need to map temporarily the memory to
a cpu-accessible memory (in our case, handled as a numpy array).

```python
with ren.mapped(x) as map:
    np.copyto(map, np.random.rand(*map.shape))
```

The context opened using `mapped(...)` function allows to map and unmap internally the
opencl array. That means that the transfer to the GPU only occurs after exiting the 
context.

## Logic (Kernels)

In `rendertoy` the logic to transform data (vertices, textures, image synthesis) is 
intended to exploit parallelization when possible. One of the main characteristics of 
the parallel programming is that logics are expressed as a SIMD architecture (Single
instruction, multiple data).

In order to express what occurs with/to a single data, we can write a kernel. A kernel
is a function (with arguments) that is executed in parallel a high number of invocations
(threads).

The main goal of a kernel is to deal with the task of a specific thread. Common 
synchronization logics such as barriers are possible but with a limit context 
(e.g., a group of threads). The organization of this threads in groups and finally in a 
grid of groups looks for efficiency in memory accessing (for example, when sharing 
memory) and it is similar among the technologies 
(CUDA, OpenCL, DirectX, OpenGL, Vulkan) differing mostly in naming. In our `rendertoy` 
project, we preferred simplicity instead. The threads are only arranged as a sequence,
internally grouped in 1024 threads each.

A kernel is a void function that receives arguments and process the task for a specific
thread, identified by the variable `thread_id`. In next example we compute in parallel
the `sin` of all values in `x`.

First, we create a buffer to store all computed values (i.e., 100 floats).

```python
y = ren.create_buffer(100, np.float32)
```

Next, we declare the kernel main. Notice the python function is an empty function,
only the documentation of the function is used as body of the function.
The signature of the kernel is also expressed as python annotations. Special attention
to the convention of using list with a single type to express a pointer to such a type.
Valid types are numpy types, int and float. Strings are also valid but in that case 
the text should be a valid OpenCL type.

The `kernel_main` decorator is used to transform the function declaration into an
executable opencl kernel. Internally, a single program is built with all declared
elements (`kernel_main`, `kernel_function`, `kernel_struct`), so, the names can not be
reused.

```python
@ren.kernel_main
def compute(x: [np.float32], y: [np.float32]):
    """
    y[thread_id] = sin(x[thread_id]); // OpenCL has a variety of math functions as builtins
    """
```

Previous python function is converted into an OpenCL kernel similar to:

```c++
__kernel void compute(
    __global float* x, 
    __global float* y, 
    __global char* memory_pool_arg) {
    memory_pool = memory_pool_arg;
    int thread_id = get_global_id(0);
    y[thread_id] = sin(x[thread_id]); // OpenCL has a variety of math functions as builtins
}
```

*The use of a memory pool (a chunk of chars) is useful later to handle dynamic objects.
OpenCL has a strong limitation on what can be expressed in a kernel or a function.*

In order to execute the kernel logic as 100 invocations we can use a fluent interface
proposed in `rendertoy` to emulate a generic invocation.

kernel_name [ Number of invocations (threads) ] ( kernel arguments )

For example, to execute 100 calls to the kernel with the sequence of `thread_id`s from
0..99, we can do:

```python
compute[100](x, y)
```

Finally, we can use `get` method in buffers to have a cpu-accessible copy of the buffer
as a numpy array.

```python
print(y.get())
```

The code here is exposed in the tutorial/lesson01_math.py script.



 