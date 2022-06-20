
import rendering as ren

@ren.kernel_main
def make_egg(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);
    float a = 0.65;
    float b = 0.5;

    if(u <= b) p = (float3)(u, -sqrt(a * a - u * u * a * a / (b * b)), 0);
    else p = p = (float3)(2 * b - u, sqrt(a * a - (2 * b - u) * (2 * b - u) * a * a / (b * b)), 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """

@ren.kernel_main
def make_conic1(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.2, b = 0.6;
    if(u <= a) p = (float3)(u, 0.2, 0);
    else if(u <= b) p = (float3)(u, -0.5 * u + 0.3, 0);
    else p = (float3)(0.6 - (u - 0.6) * 1.5, 0, 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """


@ren.kernel_main
def make_conic2(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.2, b = 0.6, c = 0.7;
    if(u <= a) p = (float3)(u, 0.2, 0);
    else if(u <= b)
    {
        float x = 0.2 + (u - a) * 0.25;
        float y = 4 * x - 0.6;
        p = (float3)(x, y, 0);
    }
    else if(u <= c) p = (float3)(0.3, 0.6 + (u - b), 0);
    else p = (float3)((1.0 - u), 0.7, 0);

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """

@ren.kernel_main
def make_plate(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float a = 0.3, b = 0.65, c = 0.7;
    if(u <= a) p = (float3)(u, 0.7, 0);
    else if(u <= b)
    {
        float x = 0.3 + (u - a) * 12.0 / 7.0;
        float y = x * x * 5 / 18.0 + 27.0 / 40.0;
        p = (float3)(x, y, 0);
    }
    else if(u <= c) p = (float3)(0.9 - (u - b) * 2.0, 0.9, 0);
    else
    {
        float x = 0.8 - (u - c) * 8.0 / 3.0;
        float y = x * x * 5 / 18.0 + 27.0 / 40.0 + 0.05;
        p = (float3)(x, y, 0);
    }

    // Evaluate rotation
    float4x4 rot = rotation(v * 3.141593 * 2, (float3)(0,1,0));
    float4 h = (float4)(p.x, p.y, p.z, 1.0);
    h = mul(h, rot);

    float3 n = normalize((float3)(1, 2, 0));
    float4 nh = (float4)(n, 1.0);
    nh = mul(nh, rot);

    vertices[thread_id].N = nh.xyz;
    vertices[thread_id].P = h.xyz; // update position of the mesh with computed parametric transformation
    """
