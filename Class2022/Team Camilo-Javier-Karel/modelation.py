import rendering as ren


@ren.kernel_main
def make_water_bottle(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float arr[] = {0.16, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float x1 = 0.16;
    if(u <= x1) p = (float3)(u, 0, 0);
    else if(u <= 0.8) {
        for (int i = 1; i < 7; i++) {
            float xx = arr[i];
            if(u <= xx) {
                float qx = xx - u;
                float diff = xx - arr[i - 1];
                float y = -(qx * (qx - diff))*3;
                p = (float3)(x1 + y, u - x1, 0);
                break;
            }
        }
    }
    else {
        float m = 0.26 / 0.16;
        float n = 0.9 - m * 0.16;
        float x = u - 0.8;
        float new_x = x * 0.55;
        p = (float3)(x1 - new_x, new_x * m + n, 0);
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


@ren.kernel_main
def water_bottle_cap(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    if(u <= 0.4) p = (float3)(u, 0, 0);
    else if(u <= 0.6) p = (float3)(0.4, (0.6 - u) * 1.5, 0);
    else p = (float3)(1 - u, 0.3, 0);

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
def make_alcohol_bottle(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float x1 = 0.12;
    if(u <= x1) p = (float3)(u, 0, 0);
    else if(u <= 0.7) {
        p = (float3)(x1, u - x1, 0);
    }
    else if(u <= 0.85) {
        float m = 1;
        float n = 0.7 - m * 0.12;
        float x = u - 0.7;
        float new_x = x * 0.5;
        p = (float3)(x1 - new_x, new_x * m + n, 0);
    }
    else p = (float3)(0.04, 0.65 + (u - 0.85) * 0.9, 0);

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
def glass_upper_transform(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float x1 = 0.16;
    float x2 = 0.23;
    float y2 = 0.6;
    float diff = x2 - x1;
    float m = y2 / (x2 - x1);
    if(u <= x1) p = (float3)(u, 0, 0);
    else
    {
        float aux = (u - x1) * (diff);
        p = (float3)(x1 + aux, aux * m, 0);
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



@ren.kernel_main
def glass_down_transform(vertices: [ren.MeshVertex]):
    """
    float2 uv = vertices[thread_id].C;

    float u = uv.x;
    float v = uv.y;

    float3 p = (float3)(0,0,0);

    float x1 = 0.16;
    float y2 = 0.04;
    float diff = y2;
    if(u <= x1) p = (float3)(u, 0, 0);
    else p = (float3)(x1, (u - x1) * diff, 0);


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
