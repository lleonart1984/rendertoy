# Texture Mapping in rendertoy

Texture mapping is a challange in `rendertoy` because the limitations of the OpenCL technology.
Efficient image managing in OpenCL is extremely limited and therefore, a workaround for 
texture mapping was necessary.

Our solution was to store all textures that needed to be accessible from shaders in a 
memory pool that is a heap of bytes. Then, a simple object relating the offset in the memory,
the pixel with and height is used to know how to sample a color.

For simplicity, only `float4` format is supported now. That means that the format is not 
necessary and sampling algorithm reduces considerably.

## Allocating a texture in the memory pool

To create space for a texture in the memory pool will be used:

```python
texture_memory, texture_descriptor = ren.create_texture2D(width, height)
```

Notice that two objects are returned. A memory object that can be mapped to a numpy array
to update/retrieve the texture pixels. Secondly, a texture descriptor that has the information
of the width, height and memory location where texture object starts to be used when sampling
inside a shader.

## Update texture pixels

As mentioned before, all textures elements are of type `float4`. To map the memory is only
necessary to update its texels with `float4` values, like shown in next.

```python
with ren.mapped(texture_memory) as map:
    map[:,:] = ren.make_float4(1,1,0,1) # Write yellow to all texels
    map[0,0] = ren.make_float4(1,0,0,1) # Write red to texel at 0,0
```

In the case that the texture comes from an image, then the pillow library can be used to load
the image in a numpy array first and then create the texture and update.

```python
from PIL import Image
image_for_texture = np.array(Image.open(f"{ROOT_DIR}/models/marble2.jpg"))
texture_memory, texture_descriptor = ren.create_texture2D(image_for_texture.shape[1], image_for_texture.shape[0])
with ren.mapped(texture_memory) as map:
    # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
    map = map.view(np.float32).ravel().reshape(image_for_texture.shape[0], image_for_texture.shape[1], 4)
    map[:,:,0:3] = image_for_texture/255.0   # update rgb from image
    map[:,:,3] = 1.0    # set alphas = 1.0
```

Notice that layout in a numpy array is different from images. Rows are dimension 0 of an array
but y's are dimension 1 of an image. It is better just to not mess with it and treat always
rows are rows and y's are y's.

Also, image loaded from pillow are rgb from 0..255, then a mapping to a 0..1 `float` is required.

## Sampling inside a shader

Once the texture is created, the only thing a shader needs to know is the descriptor because
the memory pool is always passed to any kernel function and it is accessible from everywhere.

With a texture descriptor we can sample a position of the texture in the form:

```c++
float4 texel = sample2D(texture_descriptor, coordinate);
```

Right now, only point sampling is supported. Bilinear sampling is easier to support (volunteers?)
but in any case would be another function, like `sample2D_linear`. Normally, sampling strategies
varies with respect to the algorithm, not with respect to the models. In other APIs, a sampler 
object is declared with all specification on how to wrap the coordinates if exceed 0..1, how to 
sample mip maps, how to interpolate, bias on the mip level, anisotropic factor, ...

## A final example

Let us define a vertex shader that computes a parametric coordinate for texture mapping 
(orthographic projection) and computes the lighting in another field.

```python
@ren.kernel_struct
class Vertex_Out:
    proj: ren.float4
    L: ren.float3  # Field used for light
    C: ren.float2  # Field used for the coordinates
```

For the fragment shader, there is now an information to pass, the texture descriptor.

```python
@ren.kernel_struct
class Materials:
    DiffuseMap: ren.Texture2D
```

Now the raster will receive a different global information for vertex processing than for 
fragment processing.

```python
vertex_shader_globals = ren.create_struct(Transforms)
fragment_shader_globals = ren.create_struct(Materials)
```

The shaders now consider the lighting in the vertex shader but modulates the lighting
with specific color provided by the texture on the fragment shader to compute the final color.

```python
@ren.kernel_function  
def transform_and_draw(
    vertex: ren.MeshVertex, info: Transforms
) -> Vertex_Out:
    """
    float3 P = vertex.P;
    float d = 0.2f + max(0.0f, dot(vertex.N, normalize((float3)(1,1,1))));
    float3 L = (float3)(d,d,d); // vertex.N * 0.5f + 0.5f; // use normals as a color for debugging purposes

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info.World); // transform with respect to world matrix
    H = mul(H, info.View);  // transform with respect to view matrix
    H = mul(H, info.Proj);  // transform with respect to projection matrix

    Vertex_Out o;
    o.proj = H;
    o.L = L;
    o.C = vertex.P.xy * 2;
    return o;
    """

@ren.kernel_function
def fragment_to_color(fragment: Vertex_Out, info: Materials) -> ren.float4:
    """
    float3 diff = sample2D(info.DiffuseMap, fragment.C).xyz;
    return (float4)(diff  * fragment.L, 1);
    """
```

Important, while updating the texture descriptor something happen. Descriptors are saved
by default as device buffers (internally a OpenCL array). They can be used directly as
arguments of a kernel if necessary. Nevertheless, in our example we defined a Materials
struct with at least one field for the DiffuseMap (can be extended in a future to specify
other material parameters and textures). This is also a buffer in the device memory.
When the Materials struct is mapped to a numpy array the descriptor turns into a numpy tuple.
In order to copy the texture_descriptor to that tuple we need to "copy" the texture_descriptor
to a numpy array first. That's why the correct update of the Materials buffer is calling the
`get` function:

```python
with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor.get()
```

The code here is exposed in the tutorial/lesson09_texture_mapping.py script.