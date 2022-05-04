from pywavefront import Wavefront
from ._core import create_buffer, mapped
from ._modeling import Mesh, MeshVertex
import numpy as np


def load_obj(path):
    obj = Wavefront(path, strict=False, create_materials=True, collect_faces=True, parse=True)
    objs = []

    for _, m in obj.meshes.items():
        mat = m.materials[0]
        v = mat.vertices
        vertex_count = len(v) // mat.vertex_size
        v = np.array(v, dtype=np.float32).reshape(vertex_count, mat.vertex_size)
        mesh_vertices = create_buffer(vertex_count, MeshVertex)
        mesh_indices = create_buffer(len(m.faces)*3, int)
        with mapped(mesh_vertices) as map:
            map: np.ndarray
            map = map.view(np.float32).reshape(vertex_count, -1)
            offset = 0
            for att in mat.vertex_format.split('_'):
                if att == 'N3F':
                    map[:, 4:7] = v[:, offset:offset+3]
                    offset += 3
                elif att == 'V3F':
                    map[:, 0:3] = v[:, offset:offset+3]
                    offset += 3
                elif att == 'T2F':
                    map[:, 8:10] = v[:, offset:offset+2]
                    offset += 2
                else:
                    raise Exception(f'Vertex format in obj {mat.vertex_format} is not supported')
            v_min = map[:, 0:3].min()
            v_max = map[:, 0:3].max()
            v_size = v_max - v_min
            max_dim = v_size.max()
            map[:,0:3] = (map[:,0:3] - v_min)/max_dim - v_size * 0.5 / max_dim

        objs.append((Mesh(mesh_vertices, mesh_indices), None))  # mesh + material

    return objs
