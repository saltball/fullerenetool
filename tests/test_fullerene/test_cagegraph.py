from ase.build import molecule

from fullerenetool.fullerene import CageGraph


def test_generate_fullerene():
    atoms = molecule("C60")
    graph = CageGraph.from_atoms(atoms)
    atoms = graph.generate_atoms(["C" for _ in range(60)], max_steps=500)
    for i in range(60):
        for j in range(i + 1, 60):
            assert atoms.get_distance(i, j) > 0.5
