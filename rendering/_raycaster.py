import typing
import pyopencl.array as cla
from ._modeling import Mesh
from ._core import kernel_main, kernel_struct, kernel_function, build_kernel_main, build_kernel_function
from ._core import float3


@kernel_struct
class BVH_AABB:
    min: float3
    max: float3
    count: int
    elements: int


@kernel_struct
class BVH_Triangle:
    v0: float3
    v1: float3
    v2: float3
    index: int
    mesh: int


class Raycaster:
    def __init__(self, models: typing.List[Mesh]):
        self.models  = models
        self._build_ads()

    def _build_ads(self):
        for m in self.models:
            m: Mesh
            triangles = m.vertices.shape[0] // 3 if m.indices is None else m.indices.shape[0] // 3

    def ray_cast(self, rays: cla.Array)->cla.Array:
        pass


