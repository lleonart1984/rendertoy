# My first rasterization process

Drawing the point cloud of a geometry introduces the first challanges in computer graphics.
In this section we will discuss the creation of a vertex buffer, apply transformations to
world space, view space and projection space. Everything inside a kernel that also renders
the points into an image.

## Vertex Buffer

A vertex buffer is an object in computer graphics commonly used to represent the vertices of a 
geometry if a triangular (or in general faceted) representation is used.

Normally, associated to a vertex we have different attributes, not only positions. Grouping
together those attributes allow us to process the vertex as a hole. For instance, a common
displacement transformation to the position depends on the normal; a lighting depends on
position, normal and color; a parametric texture mapping might depend on position, normal or 
even the viewer position.

In order to define a vertex buffer in `rendertoy` we need to declare the vertex struct type first.

```python
@ren.kernel_struct
class Vertex:
    P: ren.float3  # Position
    C: ren.float3  # Color
```

Recall this struct is for declarative purpose only, not intended to have instances in the host.
With the declared struct, we can create a buffer to hold several values of that type.

```python
num_vertex = 10000
vertex_buffer = ren.create_buffer(num_vertex, Vertex)
```

Notice here that alignment needed for `float3` is managed internally, but is something to have
in mind when the buffer is mapped to a numpy array, since all positions and colors in this example
have 4 floats instead of only 3.

Next code will fill the vertex buffer with positions from -1 to 1, and colors from 0 to 1.

```python
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
```

Notice that the random generates `double` values and cast to `float32` is necessary. Also, we
need 4 values for a position instead of 3 (because the alignment) and at the end all values
are view as `float3` type before updating the positions/colors.

Also, there is a nice slicing function in numpy that allows to slice only `"P"`, or only `"C"`, even
if in the sequence of vertices those values appear interlaced. Another way could have been to map first each vertex to 8 floats and then update values from 0..2
with the positions and from 4..6 with the colors (generating only 3 random values instead of 4).

## Transforming from model space to projection

In computer graphics it is very common to define vertices in a model space, for instance, a chair
with coordinates with respect to an origin. Then the first transformation (World transform) will
move the chair to the scene with respect to an origin of the world. Next, a view transform moves
everything with respect to the viewer, positioning the observer as center of the space and 
aligning the center of the visual frustum to the z-axis (positive or negative depending on the 
convention used, right handled or left handled). Finally, a projection transform will enclose
the visual space in a box know as normalized device coordinates (commonly from -1 to 1 every 
axis except z from 0 to 1).

We will define a struct with those 3 matrices.

```python
@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4

transform_info = ren.create_struct(Transforms)

with ren.mapped(transform_info) as map:
    map["World"] = ren.matmul(
        ren.scale(0.2), ren.rotate(np.pi/3, ren.make_float3(0, 1, 0))
    )
    map["View"] = ren.look_at(
        ren.make_float3(0, 0.3, 2),
        ren.make_float3(0, 0, 0),
        ren.make_float3(0, 1, 0),
    )
    map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)
```

Notice here that several functions for instantiating common transformations have been implemented
in `rendertoy`. Specifically, translations, scale, rotations, view transforms, and perspective 
projections. Also, the function matmul implements a matrix multiplication allowing to 
combine different transformations in a single matrix.

## The kernel

The next kernel is showing a way to raster vertices of a geometry as points in an image.
The kernel receives the image to be used to draw to, the vertex buffer as a Vertex pointer 
and the struct with the transformations.

```python
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
```

The position of the vertex is expanded to 4 components to build a homogeneous vector $H=(x,y,z,1)$.
The final component is 1 because is a position, and it is affected by translations (Normals are
commonly expanded with 0 to avoid translations but be affected by rotations).

After transform the vector $H$ by the three matrices, it has to be transformed back to 3D space.
This process can be easily done by dividing the three components $x,y,z$ by the last $w$.

Next, a clipping condition is perform to check whenever the point is inside the view frustum or 
not. Because $H$ is already in the normalized-device-coordinates space, fast evaluations of
less than or greater than is the only necessary thing.

If clipping test is passed, the point is scaled to the image size (dimensions of the image).
Notice the component `y` is used in the negative form, since in image coordinates `y` increases 
values down, while in 3D coordinate systems, `y` increases going up.

Finally, the pixel is updated with the color of the vertex.

There are different aspects of a rasterizer that were simplified here but will be addressed in
next sections. For instance, a depth comparison to solve correctly visibility.

The code here is exposed in the tutorial/lesson05_drawing_points.py script.





