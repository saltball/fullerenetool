import ase
import networkx as nx
import numpy as np
from ase.build import molecule

from fullerenetool.fullerene.cage import FullereneCage
from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    addons_to_fullerene,
)
from fullerenetool.operator.fullerene.addon_generator import generate_addons_and_filter


def test_generate_fullerene_derivatives():
    C60 = molecule("C60")
    addon = DerivativeGroup(
        atoms=ase.Atoms(
            "XCl",
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
    fullereneinit = FullereneCage(C60)
    dev_groups = [addon] * 2
    dev_graph, dev_fullerenes = addons_to_fullerene(
        dev_groups,
        [0, 1],
        fullereneinit,
        nx.adjacency_matrix(fullereneinit.graph.graph).todense(),
    )


def test_generate_addon_position():
    C60 = molecule("C60")
    addon = DerivativeGroup(
        atoms=ase.Atoms(
            "XCl",
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
    fullereneinit = FullereneCage(C60)
    dev_groups = [addon] * 0
    dev_graph, dev_fullerenes = addons_to_fullerene(
        dev_groups,
        [],
        fullereneinit,
        nx.adjacency_matrix(fullereneinit.graph.graph).todense(),
    )
    devgraph = DerivativeFullereneGraph(
        adjacency_matrix=dev_graph.adjacency_matrix,
        cage_elements=dev_graph.node_elements,
        addons=[],
    )
    for idx, candidategraph in enumerate(generate_addons_and_filter(devgraph, 2)):
        graph = candidategraph[1]
        print(graph)
        # dev_fullerene = DerivativeFullereneGraph(
        #     adjacency_matrix=nx.adjacency_matrix(graph).todense(),
        #     cage_elements=devgraph.node_elements,
        #     addons=[addon]*2,
        # ).generate_atoms_with_addons(
        #     algorithm="cagethenaddons",
        #     init_pos=fullereneinit.atoms.positions,
        #     check=False,
        #     use_gpu=False,
        # )
