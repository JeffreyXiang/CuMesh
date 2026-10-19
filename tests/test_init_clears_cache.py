"""Re-initializing a CuMesh must not reuse connectivity cached for the previous mesh."""
import torch
import cumesh

CUBE_V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
CUBE_F = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
          [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]


def cube(offset):
    v = torch.tensor(CUBE_V, dtype=torch.float32) + torch.tensor(offset, dtype=torch.float32)
    return v, torch.tensor(CUBE_F, dtype=torch.int32)


def two_cubes_sharing_an_edge():
    v0, f0 = cube([0, 0, 0])
    v1, f1 = cube([1, 1, 0])
    f = torch.cat([f0, f1 + 8]).long()
    remap = torch.arange(16)
    remap[8], remap[12] = 2, 6  # weld the second cube's edge onto the first cube's edge (1,1,0)-(1,1,1)
    return torch.cat([v0, v1]), remap[f].int()


def repaired(mesh, v, f):
    mesh.init(v.cuda(), f.cuda())
    mesh.repair_non_manifold_edges()
    return [t.cpu() for t in mesh.read()]


def test_reinit_matches_fresh_object():
    v, f = two_cubes_sharing_an_edge()
    fresh_v, fresh_f = repaired(cumesh.CuMesh(), v, f)

    reused = cumesh.CuMesh()
    other_v, other_f = cube([5, 5, 5])
    reused.init(other_v.cuda(), other_f.cuda())
    reused.get_edge_face_adjacency()  # cache connectivity for the first mesh
    reused_v, reused_f = repaired(reused, v, f)

    assert reused_v.shape == fresh_v.shape  # without the fix: 44 vertices instead of 16
    assert torch.equal(reused_f, fresh_f)
    assert torch.equal(reused_v, fresh_v)
