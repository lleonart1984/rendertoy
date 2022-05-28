from enum import IntEnum
from ._core import build_kernel_main, build_kernel_function, float2, float4, __ctx__, w_image2d_t, create_buffer, make_float2, int2, make_int2
import inspect
import pyopencl.tools as cltools
import numpy as np


class FillMode(IntEnum):
    NONE = 0
    POINTS = 1
    WIREFRAME = 2
    SOLID = 3


__VERTEX_PROCESS_CACHE__ = { }
__FRAGMENT_PROCESS_CACHE__ = { }
__PRIMITIVE_ASSEMBLY_CACHE__ = { }
__HOMOGENIZATION_CACHE__ = { }
__RASTER_CACHE__ = { }
__TILING_CACHE__ = { }
__INTERPOLATORS_2__ = { }
__INTERPOLATORS_3__ = { }


def _resolve_interpolators_2(vertex_type):
    cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
    if cl_vertex_name not in __INTERPOLATORS_2__:
        interpolations = '\n'.join(f"v_out.{field} = v0.{field} * (1 - alpha) + v1.{field} * alpha;" for field in vertex_type.fields)
        k = build_kernel_function(
            name=f'interpolate2_{cl_vertex_name}',
            arguments={ 'v0': vertex_type, 'v1': vertex_type, 'alpha': np.float32 },
            return_type=vertex_type,
            body=f"""
            {cl_vertex_name} v_out;
            {interpolations}
            return v_out;
            """
        )
        __INTERPOLATORS_2__[cl_vertex_name] = k
    return __INTERPOLATORS_2__[cl_vertex_name]


def _resolve_interpolators_3(vertex_type):
    cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
    if cl_vertex_name not in __INTERPOLATORS_3__:
        interpolations = '\n'.join(f"v_out.{field} = v0.{field} * (1 - alpha.x - alpha.y) + v1.{field} * alpha.x + v2.{field} * alpha.y;" for field in vertex_type.fields)
        k = build_kernel_function(
            name=f'interpolate3_{cl_vertex_name}',
            arguments={ 'v0': vertex_type, 'v1': vertex_type, 'v2': vertex_type, 'alpha': float2 },
            return_type=vertex_type,
            body=f"""
            {cl_vertex_name} v_out;
            {interpolations}
            return v_out;
            """
        )
        __INTERPOLATORS_3__[cl_vertex_name] = k
    return __INTERPOLATORS_3__[cl_vertex_name]




def _resolve_vertex_kernel(input_type, globals_type, output_type, shader):
    if shader.name not in __VERTEX_PROCESS_CACHE__:
        k = build_kernel_main(
            name=f'VertexProcess_{shader.name}',
            arguments={ 'in_vertices': [input_type], 'globals': globals_type, 'out_vertices': [output_type] },
            body=f"""
    out_vertices[thread_id] = {shader.name}(in_vertices[thread_id], globals);
            """
        )
        __VERTEX_PROCESS_CACHE__[shader.name] = k
    return __VERTEX_PROCESS_CACHE__[shader.name]


def _resolve_fragment_kernel(input_type : np.dtype, globals_type, shader):
    if shader.name not in __FRAGMENT_PROCESS_CACHE__:
        projection_field = [k for k, (d, offset) in input_type.fields.items() if offset == 0][0]

        depth_test = build_kernel_main(
            name=f'DepthTest_{shader.name}',
            arguments={ 'in_fragments': [input_type], 'globals': globals_type, 'render_target': w_image2d_t, 'depth_buffer': [np.uint32] },
            body=f"""
    float4 proj = in_fragments[thread_id].{projection_field};
    if (proj.z < 0)
        return;
    int2 dim = get_image_dim(render_target);
    int px = (int)proj.x;
    int py = (int)proj.y;
    uint depth = as_uint(proj.z);
    atomic_min(depth_buffer+(py * dim.x + px), depth);
            """
        )

        k = build_kernel_main(
            name=f'FragmentProcess_{shader.name}',
            arguments={'in_fragments': [input_type], 'globals': globals_type, 'render_target': w_image2d_t,
                       'depth_buffer': [np.uint32]},
            body=f"""
        float4 color = {shader.name}(in_fragments[thread_id], globals);
        float4 proj = in_fragments[thread_id].{projection_field};
        if (proj.z <= 0)
            return;
        int2 dim = get_image_dim(render_target);
        int px = (int)proj.x;
        int py = (int)proj.y;
        uint depth = as_uint(proj.z);
        uint written_depth = *(depth_buffer + py * dim.x + px);
        if (written_depth == depth) // minimum 
            write_imagef(render_target, (int2)(px, py), color);
                """
        )

        __FRAGMENT_PROCESS_CACHE__[shader.name] = depth_test, k
    return __FRAGMENT_PROCESS_CACHE__[shader.name]


def _resolve_homogenize_and_viewport(vertex_type: np.dtype):
    if vertex_type not in __HOMOGENIZATION_CACHE__:
        cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
        projection_field = [k for k, (d, offset) in vertex_type.fields.items() if offset == 0][0]
        k = build_kernel_main(
            name=f'Dehomogenize_{cl_vertex_name}',
            arguments={'vertex_buffer': [vertex_type], 'viewport_dim': float2 },
            body=f"""
            vertex_buffer[thread_id].{projection_field}.xyz /= vertex_buffer[thread_id].{projection_field}.w;
            vertex_buffer[thread_id].{projection_field}.y *= -1.0f;
            vertex_buffer[thread_id].{projection_field}.xy += (float2)(1, 1);
            vertex_buffer[thread_id].{projection_field}.xy *= viewport_dim * 0.5f;
                    """
        )
        __HOMOGENIZATION_CACHE__[vertex_type] = k
    return __HOMOGENIZATION_CACHE__[vertex_type]


def _resolve_primitive_assembly_and_clipping_z0(vertex_type: np.dtype):
    if vertex_type not in __PRIMITIVE_ASSEMBLY_CACHE__:
        cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
        projection_field = [k for k, (d, offset) in vertex_type.fields.items() if offset == 0][0]
        point_kernel = build_kernel_main(
            name=f'PointAssembly_{cl_vertex_name}',
            arguments={'vertex_buffer': [vertex_type], 'index_buffer': [np.int32], 'count': [np.int32], 'primitive_buffer': [vertex_type]},
            body=f"""
            int index = index_buffer == 0 ? thread_id : index_buffer[thread_id];
            float4 proj = vertex_buffer[index].{projection_field};
            if (proj.z < 0) // clipping_z0
                return;
            int out_index = atomic_add(count, 1);
            primitive_buffer[out_index] = vertex_buffer[index];
                    """
        )
        triangle_kernel = build_kernel_main(
            name=f'TriangleAssembly_{cl_vertex_name}',
            arguments={'vertex_buffer': [vertex_type], 'index_buffer': [np.int32], 'count': [np.int32], 'primitive_buffer': [vertex_type]},
            body=f"""
            int index0 = index_buffer == 0 ? thread_id*3+0 : index_buffer[thread_id*3+0];
            int index1 = index_buffer == 0 ? thread_id*3+1 : index_buffer[thread_id*3+1];
            int index2 = index_buffer == 0 ? thread_id*3+2 : index_buffer[thread_id*3+2];
            float4 proj0 = vertex_buffer[index0].{projection_field};
            float4 proj1 = vertex_buffer[index1].{projection_field};
            float4 proj2 = vertex_buffer[index2].{projection_field};
            
            int clip_mode = (proj0.z < 0 ? 1 : 0) | (proj1.z < 0 ? 2 : 0) | (proj2.z < 0 ? 4 : 0);  
            
            if (clip_mode == 7) return; // the hole triangle is back

            {cl_vertex_name} v0 = vertex_buffer[index0];
            {cl_vertex_name} v1 = vertex_buffer[index1];
            {cl_vertex_name} v2 = vertex_buffer[index2];
            
            {cl_vertex_name} v01 = interpolate2_{cl_vertex_name}(v0, v1, -proj0.z / (proj1.z - proj0.z));
            {cl_vertex_name} v12 = interpolate2_{cl_vertex_name}(v1, v2, -proj1.z / (proj2.z - proj1.z));
            {cl_vertex_name} v20 = interpolate2_{cl_vertex_name}(v2, v0, -proj2.z / (proj0.z - proj2.z));
            
            {cl_vertex_name} v_out[3];
            // First triangle
            switch (clip_mode){{
                case 0: v_out[0] = v0; v_out[1] = v1; v_out[2] = v2; break;
                case 1: v_out[0] = v01; v_out[1] = v1; v_out[2] = v2; break;
                case 2: v_out[0] = v0; v_out[1] = v01; v_out[2] = v12; break;
                case 3: v_out[0] = v12; v_out[1] = v2; v_out[2] = v20; break;
                case 4: v_out[0] = v0; v_out[1] = v1; v_out[2] = v12; break;
                case 5: v_out[0] = v01; v_out[1] = v1; v_out[2] = v12; break;
                case 6: v_out[0] = v0; v_out[1] = v01; v_out[2] = v20; break;
            }}
            
            int out_index = atomic_add(count, 1);
            primitive_buffer[3*out_index + 0] = v_out[0];
            primitive_buffer[3*out_index + 1] = v_out[1];
            primitive_buffer[3*out_index + 2] = v_out[2];

            // Second triangle
            switch (clip_mode){{
                case 1: v_out[0] = v01; v_out[1] = v2; v_out[2] = v20; break;
                case 2: v_out[0] = v0; v_out[1] = v12; v_out[2] = v2; break;
                case 4: v_out[0] = v0; v_out[1] = v12; v_out[2] = v20; break;
                default:
                return; // no second triangle needed
            }}
            out_index = atomic_add(count, 1);
            primitive_buffer[3*out_index + 0] = v_out[0];
            primitive_buffer[3*out_index + 1] = v_out[1];
            primitive_buffer[3*out_index + 2] = v_out[2];
                    """
        )
        __PRIMITIVE_ASSEMBLY_CACHE__[vertex_type] = (point_kernel, None, triangle_kernel)
    return __PRIMITIVE_ASSEMBLY_CACHE__[vertex_type]


def _resolve_raster_kernels(vertex_type: np.dtype):
    if vertex_type not in __RASTER_CACHE__:
        cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
        projection_field = [k for k, (d, offset) in vertex_type.fields.items() if offset == 0][0]
        point_raster_kernel = build_kernel_main(
            name=f'PointRaster_{cl_vertex_name}',
            arguments={'primitive_buffer': [vertex_type], 'count': [np.int32],
                       'fragment_buffer': [vertex_type]},
            body=f"""
                    int index = thread_id;
                    float4 proj = primitive_buffer[index].{projection_field};
                    if (proj.x < -proj.w || proj.x > proj.w || proj.y < -proj.w || proj.y > proj.w) // clipping
                        return;
                    int out_index = atomic_add(count, 1);
                    fragment_buffer[out_index] = primitive_buffer[index];
                            """
        )
        triangle_raster_kernel = build_kernel_main(
            name=f'TriangleRaster_{cl_vertex_name}',
            arguments={'primitive_buffer': [vertex_type], 'primitive_drawn': [np.int32], 'counting': [np.int32], 'fragment_count': [np.int32],
                       'fragment_buffer': [vertex_type], 'fragment_buffer_capacity': np.int32, 'viewport_dim': int2},
            body=f"""
                            int index = thread_id;
                            float4 proj0 = primitive_buffer[3*index + 0].{projection_field};
                            float4 proj1 = primitive_buffer[3*index + 1].{projection_field};
                            float4 proj2 = primitive_buffer[3*index + 2].{projection_field};
                            if (proj0.z < 0) return; // already rendered
                            int startx = max(0, (int)min(proj0.x, min(proj1.x, proj2.x)));
                            int starty = max(0, (int)min(proj0.y, min(proj1.y, proj2.y)));
                            int endx = min(viewport_dim.x - 1, 1 + (int)max(proj0.x, max(proj1.x, proj2.x)));
                            int endy = min(viewport_dim.y - 1, 1 + (int)max(proj0.y, max(proj1.y, proj2.y)));
                            int pixel_count = (endx - startx + 1)*(endy - starty + 1);
                            
                            int potential_index = atomic_add(counting, pixel_count);

                            if (potential_index + pixel_count < fragment_buffer_capacity)
                            {{
                                atomic_add(primitive_drawn, 1);
                                
                                {cl_vertex_name} v1 = primitive_buffer[3*index + 0];
                                {cl_vertex_name} v2 = primitive_buffer[3*index + 1];
                                {cl_vertex_name} v3 = primitive_buffer[3*index + 2];
                                
                                float2 a = v1.{projection_field}.xy;
                                float2 b = v2.{projection_field}.xy;
                                float2 c = v3.{projection_field}.xy;
                                
                                float2 vec1 = b - a;
                                float2 vec2 = c - a;
                                bool is_counter_clockwise = (vec1.x * vec2.y - vec1.y * vec2.x) <= 0;
                                
                                if (!is_counter_clockwise) // Grant is Counterclockwised
                                {{
                                    {cl_vertex_name} temp = v2;
                                    v2 = v3;
                                    v3 = temp;
                                }}
                                
                                float4 h1 = v1.{projection_field};
                                float4 h2 = v2.{projection_field};
                                float4 h3 = v3.{projection_field};
                                
                                float a1, b1, c1, a2, b2, c2, a3, b3, c3;

                                a1 = (h2.y - h1.y);
                                b1 = (h1.x - h2.x);
                                c1 = h1.x * (h1.y - h2.y) - h1.y * (h1.x - h2.x);
                
                                a2 = (h3.y - h2.y);
                                b2 = (h2.x - h3.x);
                                c2 = h2.x * (h2.y - h3.y) - h2.y * (h2.x - h3.x);
                
                                a3 = (h1.y - h3.y);
                                b3 = (h3.x - h1.x);
                                c3 = h3.x * (h3.y - h1.y) - h3.y * (h3.x - h1.x);
                
                                bool v1v2IsTLE = (h1.y == h2.y && h2.x <= h1.x) || h1.y < h2.y;
                                bool v2v3IsTLE = (h2.y == h3.y && h3.x <= h2.x) || h2.y < h3.y;
                                bool v3v1IsTLE = (h3.y == h1.y && h1.x <= h3.x) || h3.y < h1.y;
                
                                float comp3 = v1v2IsTLE ? 0 : 0.00000001;
                                float comp1 = v2v3IsTLE ? 0 : 0.00000001;
                                float comp2 = v3v1IsTLE ? 0 : 0.00000001;

                                if (pixel_count < 64*64)
                                for (int row = starty; row <= endy; row++)
                                    for (int col = startx; col <= endx; col++)
                                    {{
                                        float px = col + 0.5f;
                                        float py = row + 0.5f; // set at the middle of the pixel

                                        float d1 = a1 * px + b1 * py + c1;
                                        float d2 = a2 * px + b2 * py + c2;
                                        float d3 = a3 * px + b3 * py + c3;

                                        float alpha3 = d1 / (d1 + d2 + d3);
                                        float alpha1 = d2 / (d1 + d2 + d3);
                                        float alpha2 = d3 / (d1 + d2 + d3);
                                        
                                        float4 h_int = h1 * alpha1 + h2 * alpha2 + h3 * alpha3;

                                        if (alpha1 >= comp1 && alpha2 >= comp2 && alpha3 >= comp3)
                                        {{ // interior
                                            float beta1 = (alpha1 / h1.w) / (alpha1 / h1.w + alpha2 / h2.w + alpha3 / h3.w);
                                            float beta2 = (alpha2 / h2.w) / (alpha1 / h1.w + alpha2 / h2.w + alpha3 / h3.w);
                                            float beta3 = (alpha3 / h3.w) / (alpha1 / h1.w + alpha2 / h2.w + alpha3 / h3.w);
                                            
                                            {cl_vertex_name} fragment = interpolate3_{cl_vertex_name}(v1, v2, v3, (float2)(beta2, beta3));
                                            fragment.{projection_field} = h_int;
                                            int fragment_pos = atomic_add(fragment_count, 1);
                                            fragment_buffer[fragment_pos] = fragment;
                                        }}
                                    }}
                                // mark primitive as drawn
                                primitive_buffer[3*index + 0].{projection_field}.z = -1;
                            }}
                                    """
        )

        __RASTER_CACHE__[vertex_type] = point_raster_kernel, None, triangle_raster_kernel  # Point, Line, Triangle rasters
    return __RASTER_CACHE__[vertex_type]


# def _resolve_tiling_kernel(vertex_type: np.dtype):
#     if vertex_type not in __TILING_CACHE__:
#         cl_vertex_name = cltools.dtype_to_ctype(vertex_type)
#         projection_field = [k for k, (d, offset) in vertex_type.fields.items() if offset == 0][0]
#         triangle_tiling_kernel = build_kernel_main(
#             name=f'Tiling_{cl_vertex_name}',
#             arguments={'primitive_buffer': [vertex_type], 'count': [np.int32],
#                        'fragment_buffer': [vertex_type]},
#             body=f"""
#                     int index = thread_id;
#                     float4 proj = primitive_buffer[index].{projection_field};
#                     if (proj.x < -proj.w || proj.x > proj.w || proj.y < -proj.w || proj.y > proj.w) // clipping
#                         return;
#                     int out_index = atomic_add(count, 1);
#                     fragment_buffer[out_index] = primitive_buffer[index];
#                             """
#         )
#         __TILING_CACHE__[vertex_type] = triangle_tiling_kernel
#     return __TILING_CACHE__[vertex_type]

class Raster:

    def __init__(self, render_target, vertex_shader, vertex_shader_globals, fragment_shader, fragment_shader_globals):
        self._render_target = render_target
        self._depth_buffer = create_buffer(render_target.width * render_target.height, np.uint32)
        self._fill_mode = FillMode.WIREFRAME

        assert len(vertex_shader.signature) == 2 and vertex_shader.return_annotation is not None, "Vertex shader signature incorrect. Must receive one argument with vertex type and another with globals type, and return another struct"
        assert len(fragment_shader.signature) == 2 and fragment_shader.return_annotation==float4, "Fragment shader signature incorrect. Must receive one argument with fragment type and another with globals type, and return a float4"
        self.vertex_input_type = vertex_shader.signature[0][1].annotation
        vertex_globals_type = vertex_shader.signature[1][1].annotation
        self.vertex_output_type = vertex_shader.return_annotation
        _resolve_interpolators_2(self.vertex_output_type)
        _resolve_interpolators_3(self.vertex_output_type)
        assert fragment_shader.signature[0][1].annotation == self.vertex_output_type, "Vertex shader output must be the same type than fragment shader input."
        fragment_globals_type = fragment_shader.signature[1][1].annotation
        self.vertex_kernel = _resolve_vertex_kernel(self.vertex_input_type, vertex_globals_type, self.vertex_output_type, vertex_shader)
        self.depth_test, self.fragment_kernel = _resolve_fragment_kernel(self.vertex_output_type, fragment_globals_type, fragment_shader)
        self.vertex_shader_globals = vertex_shader_globals
        self.fragment_shader_globals = fragment_shader_globals
        self.homogenization = _resolve_homogenize_and_viewport(self.vertex_output_type)
        self.point_primitive, self.line_primitive, self.triangle_primitive = _resolve_primitive_assembly_and_clipping_z0(self.vertex_output_type)
        self.point_raster, self.line_raster, self.triangle_raster = _resolve_raster_kernels(self.vertex_output_type)

        # Buffers for streaming
        self.fragments_capacity = 32 * self._render_target.width * self._render_target.height
        self.out_fragment_buffer = create_buffer(self.fragments_capacity, self.vertex_output_type)
        self.primitive_capacity = 200000
        self.out_vertex_buffer = create_buffer(self.primitive_capacity * 3, self.vertex_output_type)
        self.out_primitive_buffer = create_buffer(self.primitive_capacity * 2 * 3, self.vertex_output_type)


    def get_render_target(self):
        return self._render_target

    def get_depth_buffer(self):
        return self._depth_buffer

    @property
    def fill_mode(self) -> FillMode:
        return self._fill_mode

    @fill_mode.setter
    def fill_mode(self, value: FillMode):
        self._fill_mode = value

    def draw_points(self, vertex_buffer, index_buffer = None):
        primitive_count = vertex_buffer.shape[0] if index_buffer is None else index_buffer.shape[0]
        max_primitive_out = primitive_count
        out_vertex_buffer = create_buffer(primitive_count, self.vertex_output_type)
        self.vertex_kernel[primitive_count](vertex_buffer, self.vertex_shader_globals, out_vertex_buffer)
        out_primitive_buffer = create_buffer(max_primitive_out, self.vertex_output_type)
        out_counter = create_buffer(1, np.int32)
        self.point_primitive[primitive_count](out_vertex_buffer, index_buffer, out_counter, out_primitive_buffer)
        visible_primitives = int(out_counter.map_to_host())
        out_fragments = create_buffer(1, np.int32)
        out_fragment_buffer = create_buffer(max_primitive_out, self.vertex_output_type)
        self.point_raster[visible_primitives](out_primitive_buffer, out_fragments, out_fragment_buffer)
        out_fragments = int(out_fragments.map_to_host())
        self.homogenization[out_fragments](out_fragment_buffer, make_float2(self._render_target.width, self._render_target.height))  # inplace modification
        self.depth_test[out_fragments](out_fragment_buffer, self.fragment_shader_globals, self._render_target, self._depth_buffer)
        self.fragment_kernel[out_fragments](out_fragment_buffer, self.fragment_shader_globals, self._render_target, self._depth_buffer)

    def draw_triangles(self, vertex_buffer, index_buffer):
        # TODO: SPLIT IN BATCHES! DO NOT SEND MORE THAN 200000 VERTICES
        # TODO: Implement fragment tiling.
        primitive_count = (vertex_buffer.shape[0] if index_buffer is None else index_buffer.shape[0])//3
        max_primitive_out = primitive_count * 2  # every triangle can be split with z=0 plane
        self.vertex_kernel[primitive_count * 3](vertex_buffer, self.vertex_shader_globals, self.out_vertex_buffer)
        out_counter = create_buffer(1, np.int32)
        self.triangle_primitive[primitive_count](self.out_vertex_buffer, index_buffer, out_counter, self.out_primitive_buffer)
        visible_primitives = int(out_counter.map_to_host())
        self.homogenization[visible_primitives*3](self.out_primitive_buffer,
                                                  make_float2(self._render_target.width, self._render_target.height))  # inplace modification
        drawn_primitives = create_buffer(1, np.int32)
        while (int(drawn_primitives.map_to_host()) < visible_primitives):
            counting = create_buffer(1, np.int32)
            out_fragments = create_buffer(1, np.int32)
            self.triangle_raster[visible_primitives](self.out_primitive_buffer, drawn_primitives, counting, out_fragments,
                                                     self.out_fragment_buffer, self.fragments_capacity, make_int2(self._render_target.width, self._render_target.height))
            out_fragments = int(out_fragments.map_to_host())
            self.depth_test[out_fragments](self.out_fragment_buffer, self.fragment_shader_globals, self._render_target,
                                                self._depth_buffer)
            self.fragment_kernel[out_fragments](self.out_fragment_buffer, self.fragment_shader_globals, self._render_target,
                                                self._depth_buffer)




