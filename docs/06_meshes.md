# Meshes

A very common object in computer graphics is the concept of Mesh. Geometries can be expressed
in different ways, but triangular representations of surfaces is one of the most efficient
to render. With a triangular representation, only vertices needs to be stored and interior
points can be interpolated from the vertices of the triangle. When this process is implemented
in screen space efficiently, is named rasterization, because consecutive pixels might share
similar attributes and therefore all pixel covered by a triangle can appear by a simple raster.

On the other hand, different triangles might share a common vertex because during 
triangulation a smooth surface was converted to a serial of triangles, a common vertex is most
of the time, the same point in the surface with same attributes. To represent this topological
information apart from the vertex information, an index buffer is used. This way a reduced number
of vertices is store in a vertex buffer while an index buffer refers to them forming triangles.

This composition of vertices and indices is commonly known as a Mesh. Some operations on
meshes include: clone, transform, subdivide, simplify, weld, compute normals, compute tangents,
and CSG operations.

In `rendertoy` there is a definition for meshes in the type `Mesh`. The operations are still
without implementation (volunteers?).

## Mesh Vertex

In order to simplify the mesh definition a standard vertex definition will be used in the 
project.

```python
@kernel_struct
class MeshVertex:
    P: float3
    N: float3
    C: float2
    T: float3
    B: float3
```

The vertex fields represents:

**Position:**  A `float3` indicating the coordinates of a position in the space of the vertex.

**Normal:** A `float3` indicating the coordinates of a normal to the surface in that position.
Normals are used when lighting (perpendicular lights reflects more light than tangent ones), 
also in subdivision strategies (subdivision needs to have a constraint on the gradients of the 
surface to grant smoothness), and also in tessellation with displacement maps (optimal 
direction to get away from the surface).

**Coordinates:**  A `float2` indicating a texture map to the surface. This coordinates allow
to know the mapping between surface points, and an image used as texture. 

**Tangent and Binormal:** Two `float3` indicating 2 tangent vectors to the surface. Normally
they are computed as the gradients where the coordinates of the texture move exclusively the axis 
`x` (tangent) or axis `y` (binormal). These vectors allows to create (in addition to the normal)
a reference frame known as Tangent Space. The space allows to transform points from model space
to the texture and vice versa. Mandatory to implement techniques such as Normal Mapping, Bump 
Mapping and Parallax Mapping.


## Loading a mesh from Obj

When a scene is designed, we model geometries (commonly with meshes) but also the materials
will affect the appearance of the surfaces for such meshes (is wood, marble, glass, metal?).
Additionally, some designers could represent groups of meshes as a single object with common
transforms (hierarchy of transformations). Specify other scene objects like lights, environment
maps, light maps, among others.

OBJ is a format defined by wavefront that is supported in `rendertoy` for its simplicity.
In current implementation, only meshes are retrieved from the files. 
The combination of a mesh with a material will be referred as "Visual" (it is not so common 
but WPF used it, Object is also used, but a little ambiguous when used in a general programming
language).

In a OBJ file could coexist several visuals. 

```python
visuals = ren.load_obj(f"{ROOT_DIR}/models/dragon.obj")
mesh, material = visuals[0]
vertex_buffer = mesh.vertices
index_buffer = mesh.indices
```

The code here is exposed in the tutorial/lesson06_loading_obj.py script.


