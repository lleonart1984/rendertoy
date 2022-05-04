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
    index_count = slices * stacks * 6
    vertices = create_buffer(vertex_count, MeshVertex)
    indices = create_buffer(index_count, np.int32)
    with mapped(vertices) as map:
        us = np.arange(0.5/slices, 1.0 + 0.5/slices, 1.0/slices)
        vs = np.arange(0.5/stacks, 1.0 + 0.5/slices, 1.0/stacks)
        pos = cartesian_product([np.array([0.0]), vs, us])
        pos = np.concatenate([pos[:,2:3], pos[:,1:2], pos[:, 0:1]], axis=-1)
        map[:, 0:3] = pos
        map[:, 8:10] = pos[:, 0:2]
    with mapped(indices) as map:
        ids = np.arange(0, slices, 1)
        for s in range(stacks):
            c00 = ids + s
            c01 = ids + 1
            c10 = ids + (slices + 1)
            c11 = ids + (1 + slices + 1)
            map[s*slices*6:s*slices*6 + s*slices*3] = np.concatenate([c00, c01, c11], axis=-1).ravel()
            map[s*slices*6 + s*slices*3:(s+1)*slices*6] = np.concatenate([c00, c11, c10], axis=-1).ravel()
    return Mesh (vertices, indices)



