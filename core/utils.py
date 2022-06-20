import rendering as ren
import numpy as np

@ren.kernel_struct
class Transforms:
    World: ren.float4x4
    View: ren.float4x4
    Proj: ren.float4x4


@ren.kernel_struct
class Materials:
    DiffuseMap: ren.Texture2D


@ren.kernel_struct
class Vertex_Out:
    proj: ren.float4
    L: ren.float3
    C: ren.float2


@ren.kernel_function  # Vertex shaders will be treated as kernel functions, the main function is implemented in the rasterizer
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
    o.C = vertex.C;
    return o;
    """

@ren.kernel_function
def fragment_to_color(fragment: Vertex_Out, info: Materials) -> ren.float4:
    """
    float3 diff = sample2D(info.DiffuseMap, fragment.C).xyz;
    return (float4)(diff * fragment.L, 1);
    """

@ren.kernel_main
def transform(vertices: [ren.MeshVertex], info: ren.float4x4):
    """
    float3 P = vertices[thread_id].P;

    float4 H = (float4)(P.x, P.y, P.z, 1.0); // extend 3D position to a homogeneous coordinates

    H = mul(H, info);

    H.xyz /= H.w; // De-homogenize
    float4 n = (float4)(vertices[thread_id].N, 1.0);
    n = mul(n, info);

    vertices[thread_id].N = normalize(n.xyz);
    vertices[thread_id].P = H.xyz;
    """

def create_and_map_textures(image):
    texture_memory, texture_descriptor = ren.create_texture2D(image.shape[1], image.shape[0])
    with ren.mapped(texture_memory) as map:
        # next change the numpy array from shape (h, w) of float4 to (h, w, 4) of floats
        map = map.view(np.float32).ravel().reshape(image.shape[0], image.shape[1], 4)
        map[:, :, 0:3] = image / 255.0  # update rgb from image
        map[:, :, 3] = 1.0  # set alphas = 1.0
    return texture_memory, texture_descriptor
