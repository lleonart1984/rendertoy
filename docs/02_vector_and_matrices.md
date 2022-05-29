# Vectors and Matrices in Rendertoy

Vectors and matrices are fundamentals in computer graphics. The use of opencl
as backend technology helps to natively and efficiently operate on vectors.
Types such as `float2`, `float3`, `int4` are builtins and also common operations
and component-wise functions.

Also, we can use component accessing to facilitate the task of creating a vector
by means of another. For instance:
```c++
float4 a = (float4)(1.0f, 0.2f, 0.3f, 1.0f);
float3 b = a.xyz;
```

Each vector type in OpenCL has a numpy equivalent in `rendertoy`, accessible through a 
type with same name (with exceptionally a `float16` that is used as a 4x4 matrix and
therefore named `float4x4`).

Another aspect to take into account is that OpenCL manages the `float3` aligned as a `float4`.
Thus, the memory of a buffer with `float3`s needs to fit 4 floats for each element.

For example, let's assume we want to transform a sequence of positions (`float3`) with a
specific transformation matrix. A transformation matrix of 4x4 is used in computer graphics
since it allows representing, not only rotations and scales, but also 
translations and projections.

Let's create a buffer with 10 initial positions, a struct to store the transformation, and
a buffer with 10 `float3` to hold the results.

```python
x = ren.create_buffer(10, ren.float3)
# Structs should be passed by value and contains only one element.
T = ren.create_struct(ren.float4x4)  
y = ren.create_buffer(x.shape[0], ren.float3)
```

If you check the numpy type behind `float3` you will realise there are three components 
`x`, `y`, `z` and a fourth unused component to handle the alignment.

Initializing values for `x` and the struct `T` can be done in several ways.
Notice the numpy type when mapping a buffer of `float3` is an unidirectional array of
type `dtype([('x','<f4'), ('y','<f4'), ('z','<f4'), ('padding','<f4')])`.

````python
with ren.mapped(x) as map:
    map.shape  # (10,)
    map.dtype  # dtype([('x','<f4'), ('y','<f4'), ('z','<f4'), ('padding','<f4')])
````

We can view the numpy array as floats, in that case:

```python
with ren.mapped(x) as map:
    map.view(np.float32).shape  # (40,)
    map.view(np.float32).dtype  # np.float32
```

Then, we can arrange the view in the desirable shape.

````python
with ren.mapped(x) as map:
    map = map.view(np.float32).reshape(-1, 4)
    map.shape   # (10, 4)
````

This way an easy way is to change the view of the numpy array to a 2D array and using
slices it can be updated only the subregion corresponding to the first 3 components. I.e.:

```python
with ren.mapped(x) as map:
    map = map.view(np.float32).reshape(-1, 4)
    map[:, 0:3] = np.random.rand(map.shape[0], 3).astype(np.float32)
```

or copying all values, like:

```python
with ren.mapped(x) as map:
    np.copyto(
        map.view(np.float32).reshape(-1, 4), 
        np.random.rand(map.shape[0], 4).astype(np.float32)
    )
```

The case of a struct is similar. The difference is that the numpy array behind a struct is a 
single scalar (in terms of numpy, in reality there are 16 values in the case of a `float4x4`). 
In those cases, copying is easier way to update.
    
```python
with ren.mapped(T) as map:
    np.copyto(map,
        ren.make_float4x4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ),
    )
```

Now, let's define a kernel that will receive the buffer with the positions (`float3*`),
the transform to apply (a single `float4x4`) and the buffer to store the transformed 
positions.

```python
@ren.kernel_main
def transform(x: [ren.float3], T: ren.float4x4, y: [ren.float3]):
    """
    float4 p = (float4)(x[thread_id], 1.0f); // expanding the float3 as a position to a homogeneous vector.
    p = mul(p, T); // apply the transform and save in same local variable
    y[thread_id] = p.xyz / p.w; // de-homogenize
    """
```

Notice the use of list annotation for the buffers `x` and `y`, but the struct `T` is passed
by value. Each invocation of the kernel will handle the transformation of the `thread_id`-th
position.

Inspecting more in details this function we can notice different OpenCL features.
First, the way a `float4` can be created by means of smaller vectors.

```c++
float4 x = (float4)(0.0f, 0.1f, 0.1f, 1.0f);
float3 y = (float3)(0.1f, 0.2f, 0.3f);
float4 z = (float4)(y, 2.0f);
```

Important, the distinction in OpenCL between `float` and `double` literals. Note the use of 
the postfix `f` after all floating-point values to declare all of them as `float`. OpenCL is strict
with all operation overloads and does not allow treating implicitly `double` as `float`.

The function `mul` is not present in OpenCL but included in the code at the beginning. Similar
with other functions like `rotation`, `translate`, `scale`, `transpose`.

Next, the signatures of such functions.

```python
float4x4 transpose( float4x4 m );
float4x4 rotation(float angle, float3 axis);
float4x4 translate(float3 v);
float4x4 scale(float3 v);
float4 mul(float4 v, float4x4 m);
```

Finally, to dispatch all threads we can call to the kernel in the form:

```python
transform[x.shape](x, T, y)
```

The use of a shape to enumerate the threads is valid, and the dispatcher uses the product
of all dimensions as the number of threads to dispatch.
For instance, if a kernel is dispatch with a tuple `(10, 50)`, 500 threads are invoked.

The code here is exposed in the tutorial/lesson02_vectors_and_matrices.py script.



