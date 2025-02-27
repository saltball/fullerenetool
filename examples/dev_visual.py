import os
from pathlib import Path

import networkx as nx
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data.colors import cpk_colors, jmol_colors
from ase.io.extxyz import read_extxyz
from ase.visualize import view
from matplotlib import pyplot as plt

from fullerenetool.fullerene.cage import BaseAbstactGraph, CageGraph, FullereneCage
from fullerenetool.fullerene.derivative_group_generator import (
    DerivativeGroupType,
    generate_substituent_coords,
)
from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    addons_to_fullerene,
)
from fullerenetool.operator.graph import canon_graph, nx_to_nauty

os.chdir(Path(__file__).parent)
cagexyz = list(read_extxyz(Path("fullerene_xyz/C80_000031924opt.xyz").open("r")))[-1]

cage = FullereneCage(cagexyz)

new_graph = canon_graph(nx_to_nauty(cage.graph.graph, include_z_labels=False))

graph = nx.from_dict_of_lists(new_graph.adjacency_dict)
cage = CageGraph(
    adjacency_matrix=nx.adjacency_matrix(graph).todense(),
    node_elements=[6 for _ in range(len(cagexyz))],
)
atoms = cage.generate_atoms(b2n2_steps=10000)
cage = FullereneCage(atoms)
XCl = DerivativeGroup(
    atoms=Atoms(
        "XCF3",
        generate_substituent_coords("XCF3", DerivativeGroupType.TETRAHEDRAL),
    ),
    graph=nx.from_numpy_array(np.array(DerivativeGroupType.TETRAHEDRAL.value["topo"])),
    addon_atom_idx=0,
)
dev_groups = [XCl] * 80

dev_graph, dev_fullerenes = addons_to_fullerene(
    dev_groups,
    list(range(80)),
    cage,
    nx.adjacency_matrix(cage.graph.graph).todense(),
    max_steps=500,
)

view(dev_fullerenes)

devgraph = DerivativeFullereneGraph(
    adjacency_matrix=dev_graph.adjacency_matrix,
    cage_elements=dev_graph.cage_elements,
    addons=dev_groups,
    atoms=dev_fullerenes,
)

dev_graph.visualize(
    atoms=dev_fullerenes,
    color="cpk",
    # color=["c" for _ in dev_graph.cage_elements] + ["r","w" for _ in dev_graph.addons],
)
