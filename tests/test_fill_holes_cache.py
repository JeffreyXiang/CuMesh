import unittest

import torch

from cumesh import CuMesh


class FillHolesCacheTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for CuMesh")

    def test_preserves_existing_topology_cache_when_no_boundary_loops(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device="cuda",
        )
        faces = torch.tensor(
            [
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
            ],
            dtype=torch.int32,
            device="cuda",
        )

        mesh = CuMesh()
        mesh.init(vertices.contiguous(), faces.contiguous())
        mesh.get_edges()
        mesh.get_boundary_info()
        mesh.get_boundary_loops()
        torch.cuda.synchronize()

        self.assertEqual(mesh.num_edges, 6)
        self.assertEqual(mesh.num_boundaries, 0)

        mesh.fill_holes(9999.0)
        torch.cuda.synchronize()

        self.assertEqual(mesh.num_edges, 6)
        self.assertEqual(mesh.num_boundaries, 0)

    def test_preserves_existing_topology_cache_when_no_holes_are_selected(self):
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            device="cuda",
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

        mesh = CuMesh()
        mesh.init(vertices.contiguous(), faces.contiguous())
        mesh.get_edges()
        mesh.get_boundary_info()
        mesh.get_boundary_loops()
        torch.cuda.synchronize()

        self.assertEqual(mesh.num_edges, 3)
        self.assertEqual(mesh.num_boundaries, 3)

        mesh.fill_holes(0.0)
        torch.cuda.synchronize()

        self.assertEqual(mesh.num_edges, 3)
        self.assertEqual(mesh.num_boundaries, 3)


if __name__ == "__main__":
    unittest.main()
