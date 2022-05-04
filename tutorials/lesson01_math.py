import lesson_common

# importing rendering functions.
import rendering as ren
import numpy as np

"""
This project is based on OpenCL and Numpy. Math capabilities are inherited from those APIs.
Programming in OpenCL is possible through a facade in rendering that encapsulates the creation of buffer,
kernels, functions and mappings.
"""

## Computing the sin of several numbers.

# Creating the buffer (an opencl buffer in the specific device)
x = ren.create_buffer(100, np.float32)
# fill the buffer with random numbers
with ren.mapped(x) as map:  # the buffer needs to be mapped to host memory
    # mapped memory is managed as a numpy array
    np.copyto(map, np.random.rand(*map.shape))

# Creating the output buffer
y = ren.create_buffer(100, np.float32)

# Creating a kernel to compute the sin
"""
In rendering, kernels are defined in a form of a function with all arguments annotated with valid types. i.e. numpy dtypes or int, or float.
Arguments that are global pointers must be specified as a list. e.g. [np.float32] represents a float* 
Intrinsicly there is a variable with the current thread id. Notice kernels will be invoke for all thread indices from 0 ... num_threads-1 when dispatch.
"""


@ren.kernel_main
def compute(x: [np.float32], y: [np.float32]):
    """
    y[thread_id] = sin(x[thread_id]); // OpenCL has a variety of math functions as builtins
    """


# Executing the kernel. name_of_function [ Number of threads ] ( arguments )
compute[100](x, y)

print(y.get())
