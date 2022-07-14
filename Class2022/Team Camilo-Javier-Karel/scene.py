import math
from modelation import *
import rendering as ren
from textures import *
import pyopencl as cl


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


def capture_scene(climage, name):
    width, height = climage.shape
    arr = np.empty((height, width, 4), dtype=np.uint8)
    cl.enqueue_copy(ren._core.__queue__, arr, climage, origin=(0, 0), region=(width, height)).wait()
    image = Image.fromarray(arr[:, :, [2, 1, 0]])
    image.save(f'{name}.png')


# Meshes

# glass
glass_upper_mesh = ren.manifold(255,255)
glass_upper_vertices =  glass_upper_mesh.vertices
glass_upper_transform[glass_upper_vertices.shape](glass_upper_vertices)
transformations = ren.to_array(ren.scale(1/4)) @ ren.to_array(ren.translate(ren.make_float3(0, 0.014, 0)))
transform[glass_upper_vertices.shape](glass_upper_vertices, transformations)


glass_down_mesh = ren.manifold(255,255)
glass_down_vertices =  glass_down_mesh.vertices
glass_down_transform[glass_down_vertices.shape](glass_down_vertices)
transformations = ren.to_array(ren.scale(1/4))
transform[glass_down_vertices.shape](glass_down_vertices, transformations)


# water bottle
water_bottle_mesh = ren.manifold(200,200)
water_bottle_vertices = water_bottle_mesh.vertices
make_water_bottle[water_bottle_vertices.shape](water_bottle_vertices)
transformations = ren.to_array(ren.scale(1/4)) @ ren.to_array(ren.translate(ren.make_float3(0.06, 0.05, -0.08)))
transform[water_bottle_vertices.shape](water_bottle_vertices, transformations)

water_bottle_cap_mesh = ren.manifold(200,200)
water_bottle_cap_vertices =  water_bottle_cap_mesh.vertices
water_bottle_cap[water_bottle_cap_vertices.shape](water_bottle_cap_vertices)
transformations = ren.to_array(transformations) @ ren.to_array(ren.scale(1/6)) @ \
                  ren.to_array(ren.translate(ren.make_float3(0.0475, 0.25, -0.05)))
transform[water_bottle_cap_vertices.shape](water_bottle_cap_vertices, transformations)


# alcohol_bottle
alcohol_bottle_mesh = ren.manifold(200,200)
alcohol_bottle_vertices = alcohol_bottle_mesh.vertices
make_alcohol_bottle[alcohol_bottle_vertices.shape](alcohol_bottle_vertices)
transformations = ren.to_array(ren.scale(1/3)) @\
                  ren.to_array(ren.translate(ren.make_float3(-0.07, 0.07, -0.08)))
transform[alcohol_bottle_vertices.shape](alcohol_bottle_vertices, transformations)


# table
table_mesh = ren.manifold(225, 225)
table = table_mesh.vertices
transformations = ren.to_array(ren.translate(ren.make_float3(-0.5, -0.5, 0))) @ \
                  ren.to_array(ren.rotate(math.pi / 2, ren.make_float3(1, 0, 0)))

transform[table.shape](table, transformations)
with ren.mapped(table) as map:
    map['N'] = ren.normalize(ren.make_float3(0, 1, 0))


# background
back_mesh = ren.manifold(200, 200)
back = back_mesh.vertices
transformations = ren.to_array(ren.translate(ren.make_float3(-0.5, -0.5, 0))) @ \
                  ren.to_array(ren.scale(5)) @ \
                  ren.to_array(ren.translate(ren.make_float3(0, 0, -2)))
transform[back.shape](back, transformations)


vertex_shader_globals = ren.create_struct(Transforms)
fragment_shader_globals = ren.create_struct(Materials)


presenter = ren.create_presenter(800, 600)

raster = ren.Raster(
    presenter.get_render_target(),  # render target to draw on
    transform_and_draw,             # vertex shader used, only transform and set normal as a color
    vertex_shader_globals,          # buffer with the transforms
    fragment_to_color,              # fragment shader, only return the color of the vertex
    fragment_shader_globals         # buffer with the material texture
)


with ren.mapped(vertex_shader_globals) as map:
    map["World"] = ren.scale(3/4)
    map["View"] = ren.look_at(
        ren.make_float3(-0.03, 0.4, 0.38),
        ren.make_float3(0, 0.1, 0),
        ren.make_float3(0, 1, 0),
    )
    map["Proj"] = ren.perspective(aspect_ratio=presenter.width / presenter.height)

ren.clear(raster.get_render_target())
ren.clear(raster.get_depth_buffer(), 1.0)

'''
with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_dark_wood.get()
raster.draw_triangles(table, table_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_glass.get()
raster.draw_triangles(glass_upper_vertices, glass_upper_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_glass_down.get()
raster.draw_triangles(glass_down_vertices, glass_down_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_bottle.get()
raster.draw_triangles(water_bottle_vertices, water_bottle_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_bottle_cap.get()
raster.draw_triangles(water_bottle_cap_vertices, water_bottle_cap_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_alcohol.get()
raster.draw_triangles(alcohol_bottle_vertices, alcohol_bottle_mesh.indices)

with ren.mapped(fragment_shader_globals) as map:
    map["DiffuseMap"] = texture_descriptor_yellow.get()
raster.draw_triangles(back, back_mesh.indices)


'''
raster.draw_points(glass_upper_vertices)
raster.draw_points(glass_down_vertices)
raster.draw_points(water_bottle_vertices)
raster.draw_points(water_bottle_cap_vertices)
raster.draw_points(alcohol_bottle_vertices)
raster.draw_points(table)
raster.draw_points(back)


presenter.present()
capture_scene(presenter.get_render_target(), 'scene_meshes')

while True:
    event, arg = presenter.poll_events()
    if event == ren.Event.CLOSED:
        break