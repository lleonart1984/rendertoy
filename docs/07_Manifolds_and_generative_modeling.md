# Manifolds and Generative Modeling

A manifold is an object that preserves the topological properties in all its domain. For instance,
a sphere is a 2-Manifold since every point of the surface of the sphere is locally similar to a 
disk.

We use the term to name meshes where all vertices reassemble a plane. In other words, all 
geometries that can be formed by wrapping a cloth without inter-crossing.

# The base surface

The call to `create_manifold(slices, stacks)` creates a mesh with vertices $(slices+1)\times(stacks+1)$
positioned from 0..1 in $x$ and $y$ coordinates. Also, coordinates have such values.
An index buffer is built to represent $slices\times\stacks\times2$ triangles. Those triangles cover
the quad and they share common vertices, so, transforming vertices will represent always a closed
surface (the cloth is never broken).

```python
mesh = ren.manifold(48, 48)
```

## The power of a rotated bezier curve

The next code represents a generative model via revolution of a bezier-curve with respect to 
a direction.

```python
@ren.kernel_function
def C_n_k(n: int, k: int) -> int:
    """
    // This function evaluates a coefficient of a Bernstein polynomial
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
```

The task of this kernel is to use the texture coordinates (from 0..1) and use them as parameters
of a generative model. Specifically, the $x$ coordinate is used as the parameter $t$ of a bezier
curve defined by some control points. The $y$ coordinate is used then as the angle for the 
revolution transformation. Finally, the position is updated with the final generated position h.

Notice in this example, the vertex buffer is sent as a pointer, and the "length" of the
array is not necessary because the `thread_id` will be in range of valid vertices. Nevertheless,
for the control points, the pointer needs to be coupled with an integer specifying the number of
control points.

By updating the control points from one frame to another, and updating the mesh "in-place" with
this kernel, the generative model can be redefined every frame and with it, to get an
animation of the model.

The code here is exposed in the tutorial/lesson07_generative_modeling.py script.

