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


# importing rendering functions.
import rendering as ren
import numpy as np

"""
OpenCL has a variety of builtin types for vectors, e.g. float2, float3, int4, etc.
Also, has a float16 that we will use as a 4x4 matrix.
In rendering an equivalent numpy type is created for each of those builtins.
"""

## Transforming a bunch of vectors

# Creating the buffer (an opencl buffer in the specific device)
x = ren.create_buffer(10, ren.float3)
T = ren.create_struct(
    ren.float4x4
)  # Structs should be passed by value and contains only one element.
y = ren.create_buffer(x.shape[0], ren.float3)

with ren.mapped(x) as map:
    np.copyto(
        map, np.random.rand(*map.shape, 4).astype(np.float32).ravel().view(ren.float3)
    )

with ren.mapped(T) as map:
    np.copyto(
        map,
        ren.make_float4x4(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )

# Creating a kernel to compute the transform
@ren.kernel_main
def transform(x: [ren.float3], T: ren.float4x4, y: [ren.float3]):
    """
    float4 p = (float4)(x[thread_id], 1.0f); // expanding the float3 as a position to a homogeneous vector.
    p = mul(p, T); // apply the transform and save in same local variable
    y[thread_id] = p.xyz / p.w; // de-homogenize
    """


# Executing the kernel. name_of_function [ Number of threads ] ( arguments )
transform[x.shape](
    x, T, y
)  # if a shape is used, the number of all axis multiplied is assumed.

print(x.get())
print(y.get())
