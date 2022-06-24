import pyopencl as cl
import pyopencl.array as cla
from enum import IntEnum
from ._core import kernel_struct, float3, float2, __queue__, create_buffer, mapped
import numpy as np


class WeldMode(IntEnum):
    NONE = 0
    POSITION = 1
    POSITION_NORMAL_TEXTURE = 2
    ALL_ATTRIBUTES = 3


class SubdivisionMode(IntEnum):
    NONE = 0
    FLAT = 1
    LOOP = 2
    BUTTERFLY = 3


@kernel_struct
class MeshVertex:
    P: float3
    N: float3
    C: float2
    T: float3
    B: float3


class Mesh:
    def __init__(self, vertices: cla.Array, indices: cla.Array):
        self.vertices = vertices
        self.indices = indices

    def clone(self) -> 'Mesh':
        raise Exception('Not implemented yet')

    def weld(self, weldMode: WeldMode, epsilon: float) -> 'Mesh':
        raise Exception('Not implemented yet')

    def simplify(self, max_vertices) -> 'Mesh':
        raise Exception('Not implemented yet')

    def subdivide(self, subdivision_mode: SubdivisionMode, **kwargs) -> 'Mesh':
        raise Exception('Not implemented yet')

    def compute_normals(self):
        raise Exception('Not implemented yet')

    def compute_tangents(self):
        raise Exception('Not implemented yet')


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def manifold(slices, stacks) -> 'Mesh':
    vertex_count = (slices + 1)*(stacks + 1)

    # I setted (stacks + 1) to make the mesh as a cilinder,
    # joining points in the last and first columns making triangles.
    # I think this is useful when making objects using revolutions
    # because this way we can add textures to the whole object,
    # the other way would be to make the last and first column to match
    # index_count = slices * (stacks + 1) * 6
    index_count = slices * stacks * 6
    vertices = create_buffer(vertex_count, MeshVertex)
    vertex_size = MeshVertex.itemsize // 4
    indices = create_buffer(index_count, np.int32)
    with mapped(vertices) as map:
        map = map.ravel().view(np.float32).reshape(-1, vertex_size)
        us = np.arange(0, 1.0 + 0.5/slices, 1.0/slices)
        vs = np.arange(0, 1.0 + 0.5/stacks, 1.0/stacks)
        pos = cartesian_product(np.array([0.0]), vs, us)
        pos = np.concatenate([pos[:,2:3], pos[:,1:2], pos[:, 0:1]], axis=-1)
        map[:, 0:3] = pos
        map[:, 8:10] = pos[:, 0:2]
    with mapped(indices) as map:
        ids = np.arange(0, slices, 1)
        for s in range(stacks): # (stacks + 1) to join first and last columns
            # The way the point indices in the columns were being calculated
            # was not correct, we need to use the number of points
            # in each column somehow.
            c00 = ids + s * slices
            c01 = ids + s * slices + 1
            c10 = ids + (s + 1) * slices
            c11 = ids + (s + 1) * slices + 1

            # np.concatenate([a,b,c]) concatenates the arrays a,b and c one after
            # the other, I think that is not the expected behavior here.
            # np.stack([a,b,c], axis=-1) zips the array elements as expected,
            # making a np.array with shape=(n,3) where n is the length of each of
            # the arrays a, b and c. np.ravel just flattens that array.
            map[s*slices*6:s*slices*6 + slices*3] = np.stack([c00, c01, c11], axis=-1).ravel()
            map[s*slices*6 + slices*3:(s+1)*slices*6] = np.stack([c00, c11, c10], axis=-1).ravel()
    return Mesh (vertices, indices)


# def parametric_transform(mesh: Mesh, f):
#     with mapped(mesh.vertices) as map:
#         map = map.ravel().view(np.float32).reshape(len(map), -1)
#         parameters = map[:, 8:10]
#         positions = f(parameters)
#         map[:, 0:3] = positions
#
#
# def generate_model(mesh: Mesh, f, g):
#     def h(uv):
#         curve = g(uv[:,0:1])
#         return f(curve, uv[:,1:2])
#     parametric_transform(mesh, h)
#
#
# def bezier_curve(t: np.ndarray, control_points: np.ndarray):
#     if control_points.dtype == float3:
#         control_points = control_points.ravel().view(np.float32).reshape(len(control_points), -1)
#     cps = np.expand_dims(control_points, axis=0).repeat(len(t), axis=0)
#     while(cps.shape[1] > 1):
#         new_cps = np.zeros(shape=(cps.shape[0], cps.shape[1] - 1, cps.shape[2]), dtype=np.float32)
#         for i in range(new_cps.shape[1]):
#             new_cps[:,i,:] = cps[:, i, :] * (1 - t) + cps[:, i+1, :] * t
#         cps = new_cps
#     return cps[:, 0, :][:, 0:3]

