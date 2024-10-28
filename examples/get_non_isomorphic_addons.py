import ase
import networkx as nx
import numpy as np
import pynauty as pn
import torch
from ase.build import molecule

from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    FullereneCage,
    addons_to_fullerene,
)
from fullerenetool.operator.fullerene.addon_generator import generate_addons_and_filter
from fullerenetool.operator.graph import nx_to_nauty

print("CUDA:", torch.cuda.is_available())


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

canon_index = pn.canon_label(
    nx_to_nauty(fulleren_init.graph.graph, include_z_labels=False)
)

fulleren_init = FullereneCage(C60[np.array(canon_index)])
print(pn.canon_label(nx_to_nauty(fulleren_init.graph.graph, include_z_labels=False)))

addon = XCl
addon_start = 1
start_idx = [1]
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
    cage_elements=dev_graph.cage_elements,
    addons=dev_groups,
)
candidategraph_list = []
candidategraph_name_list = []
atoms_list = []
addon_pos_index_list = []

cage_graph = devgraph.cage_graph
print("devgraph", devgraph)
tmp_cage_graph = cage_graph.graph.copy()
tmp_pn_graph = pn.canon_graph(nx_to_nauty(tmp_cage_graph, include_z_labels=False))
print("cage_graph", cage_graph)
for idx, candidategraph in enumerate(generate_addons_and_filter(devgraph, add_num)):
    print(idx, candidategraph)
    print(candidategraph[1].nodes)

# fulleren_init.graph.visualize()
