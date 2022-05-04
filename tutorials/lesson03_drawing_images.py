import lesson_common

import rendering as ren


# creates an image with 6 pixels width, 4 pixels height, and 4 floats (r,g,b,a)
im = ren.create_image2d(6, 4, ren.float4)

# map image memory as a numpy array with shape (4, 6, 4) height, width, components, and dtype = float32
with ren.mapped(im) as map:
    map[
        :, :, 0
    ] = 1  # change all red components to 1 as an example of what can be done with the mapped memory.
    print(map)
# outside the context region (with) the mapped memory is updated to the resource (image).


# define a kernel to apply an action for every pixel
@ren.kernel_main
def clear_image(
    im: ren.w_image2d_t,
):  # the argument is annotated with the image type. In rendering all images are treated as read_write for simplicity.
    """
    int2 dim = get_image_dim(im);
    // in rendering, only linear layout of threads is allowed. Mapping to image positions needs to be done manually.
    int px = thread_id % dim.x;
    int py = thread_id / dim.x;
    write_imagef(im, (int2)(px,py), (float4)(1.0, 0.5, 0.3, 1.0));
    """


# invoke the kernel for each pixel of the image (product of all dimensions of the image shape).
clear_image[im.shape](im)


# again map the memory to check the modified data.
with ren.mapped(im) as map:
    print(map)
