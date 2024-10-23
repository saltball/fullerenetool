import ase
import networkx as nx
import numpy as np
from ase.build import molecule

from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    FullereneCage,
    addons_to_fullerene,
)
from fullerenetool.operator.fullerene.addon_generator import generate_addons_and_filter

C60 = molecule("C60")

XCl = DerivativeGroup(
    atoms=ase.Atoms(
        "XF",
        [
            [0.1, 0.1, -0.2],
            [0, 0, 0],
            # [0, 0, 1.4]
        ],
    ),
    graph=nx.from_numpy_array(
        np.array(
            [
                [
                    0,
                    1,
                ],
                [
                    1,
                    0,
                ],
            ]
        )
    ),
    addon_atom_idx=0,
)

fulleren_init = FullereneCage(C60)
addon = XCl
addon_start = 2
start_idx = [0, 8]
add_num = 1

dev_groups = [addon] * addon_start
dev_graph, dev_fullerenes = addons_to_fullerene(
    dev_groups,
    start_idx,
    fulleren_init,
    nx.adjacency_matrix(fulleren_init.graph.graph).todense(),
)
devgraph = DerivativeFullereneGraph(
    adjacency_matrix=dev_graph.adjacency_matrix,
    cage_elements=dev_graph.node_elements,
    addons=[],
)
candidategraph_list = []
candidategraph_name_list = []
atoms_list = []
addon_pos_index_list = []

for idx, candidategraph in enumerate(generate_addons_and_filter(devgraph, add_num)):
    print(idx, candidategraph)
