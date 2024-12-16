import networkx as nx
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data.colors import cpk_colors, jmol_colors
from ase.visualize import view
from matplotlib import pyplot as plt

from fullerenetool.fullerene.cage import BaseAbstactGraph, CageGraph, FullereneCage
from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    addons_to_fullerene,
)
from fullerenetool.operator.graph import canon_graph, nx_to_nauty

C60 = molecule("C60")


cage = FullereneCage(C60)

new_graph = canon_graph(nx_to_nauty(cage.graph.graph, include_z_labels=False))

graph = nx.from_dict_of_lists(new_graph.adjacency_dict)
cage = CageGraph(
    adjacency_matrix=nx.adjacency_matrix(graph).todense(),
    node_elements=[6 for _ in range(60)],
)
C60 = cage.generate_atoms(b2n2_steps=10000)
cage = FullereneCage(C60)
XCl = DerivativeGroup(
    atoms=Atoms(
        "XHe",
        [
            [0.5, 0.5, 0.0],
            #  [0, 0, 0],
            [0, 0, 1.4],
        ],
    ),
    graph=nx.from_numpy_array(
        np.array(
            [
                [0, 1],
                [1, 0],
                # [0, 1, 0],
                # [1, 0, 1],
                # [0, 1, 0]
            ]
        )
    ),
    addon_atom_idx=0,
)
dev_groups = [XCl] * 4
dev_graph, dev_fullerenes = addons_to_fullerene(
    dev_groups,
    [0, 2, 3, 56],
    cage,
    nx.adjacency_matrix(cage.graph.graph).todense(),
)

devgraph = DerivativeFullereneGraph(
    adjacency_matrix=dev_graph.adjacency_matrix,
    cage_elements=dev_graph.cage_elements,
    addons=dev_groups,
    atoms=dev_fullerenes,
)
dev_graph.visualize(
    atoms=dev_fullerenes,
    color=["c" for _ in dev_graph.cage_elements] + ["r" for _ in dev_graph.addons],
)
