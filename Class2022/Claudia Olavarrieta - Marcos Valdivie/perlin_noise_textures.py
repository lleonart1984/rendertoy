import math
import numpy as np
from PIL import Image
import rendering as ren

@ren.kernel_function
def smoothNoise(x: np.float32, y:np.float32, noise:[np.float32], w:np.int32, h:np.int32) -> np.float32:
    '''
    float fx = x - (int)x;
    float fy = y - (int)y;

    int x1 = ((int)x + w) % w;
    int y1 = ((int)y + h) % h;
    int x2 = (x1 + w - 1) % w;
    int y2 = (y1 + h - 1) % h;

    float n1 = noise[y1 * w + x1];
    float n2 = noise[y1 * w + x2];
    float n3 = noise[y2 * w + x1];
    float n4 = noise[y2 * w + x2];

    return fx * fy * n1 + (1 - fx) * fy * n2
        + fx * (1 - fy) * n3 + (1 - fx) * (1 - fy) * n4;
    '''


@ren.kernel_function
def generateTurbulence(x:np.float32, y:np.float32, size:np.float32, noise:[np.float32], w:np.int32, h:np.int32) -> np.float32:
    '''
    float value = 0;
    float ini = size;
    while(size >= 1.0){
        value += smoothNoise(x / size, y / size, noise, w, h) * size;
        size /= 2.0;
    }
    return 128.0 * value / ini;
    '''


@ren.kernel_main
def generateMarble(noise:[np.float32], arr: ren.w_image2d_t, xperiod:np.float32,
                   yperiod:np.float32, turbpower:np.float32, turbsize:np.float32):
    '''
    int2 dim = get_image_dim(arr);
    int width = dim.x;
    int height = dim.y;

    int x = thread_id % width;
    int y = thread_id / width;

    float xy = ((float)x * xperiod / width + (float)y * yperiod / height
            + turbpower * generateTurbulence(x, y, turbsize, noise, width, height) / 256.0);
    float value =  255 * fabs(sin(3.14159 * xy));
    write_imagef(arr, (int2)(x,y), (float4)(value, value, value, 1.0));
    '''

@ren.kernel_main
def generateWood(noise:[np.float32], arr: ren.w_image2d_t,period:np.float32,
                 turbpower:np.float32, turbsize:np.float32):
    '''
    int2 dim = get_image_dim(arr);
    int width = dim.x;
    int height = dim.y;

    int x = thread_id % width;
    int y = thread_id / width;

    float xx = ((float)x - (float)width / 2.0) / width;
    float yy = ((float)y - (float)height / 2.0) / height;
    float dist = sqrt(xx * xx + yy * yy)
        + turbpower * generateTurbulence(x, y, turbsize, noise, width, height) / 256.0;
    float sine = 128.0 * fabs(sin(2 * period * dist * 3.14159));
    write_imagef(arr, (int2)(x,y), (float4)(80 + sine, 30 + sine, sine, 1.0));
    '''


def marble(width, height, xperiod:float=5.0, yperiod:float=10.0, turbpower:float=5.0, turbsize:float=32.0):
    noise = ren.create_buffer(width * height, np.float32)
    with ren.mapped(noise) as map:
        np.copyto(map.view(np.float32),
        np.random.rand(width * height).astype(np.float32))

    arr = ren.create_image2d(width, height, ren.float4)
    generateMarble[width * height](noise, arr, xperiod, yperiod, turbpower, turbsize)

    image = np.zeros((arr.shape[1], arr.shape[0], 3))
    with ren.mapped(arr) as map:
        np.copyto(image, map[:,:,:3])
    return image.astype(np.uint8)

def wood(width, height, period, turbpower, turbsize):
    noise = ren.create_buffer(width * height, np.float32)
    with ren.mapped(noise) as map:
        np.copyto(map.view(np.float32),
                  np.random.rand(width * height).astype(np.float32))

    arr = ren.create_image2d(width, height, ren.float4)
    generateWood[width * height](noise, arr, period, turbpower, turbsize)

    image = np.zeros((arr.shape[1], arr.shape[0], 3))
    with ren.mapped(arr) as map:
        np.copyto(image, map[:, :, :3])
    return image.astype(np.uint8)

#arr = marble(200, 200, 0.0, 2.0, 1.5, 32.0)
#arr = wood(200, 200, period=6.0, turbpower=0.3, turbsize=32.0)
#image = Image.fromarray(arr)
#image.show()