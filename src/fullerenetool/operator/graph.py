import ase
import networkx as nx
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from fullerenetool.logger import logger


def canon_graph(g):
    """
    NOTE: current pynauty lib is not working for graph with more than 64 nodes
    Compute the canonically labeled version of graph g.

    *g*
        A Graph object.

    return ->
        new canonical graph.
    """
    from pynauty import Graph, certificate

    if g.vertex_coloring:
        raise RuntimeError(
            "canon_graph() is not implemented for vertex-colored graphs yet."
        )
    c = certificate(g)
    set_length = len(c) // g.number_of_vertices
    sets = [
        c[set_length * k : set_length * (k + 1)] for k in range(g.number_of_vertices)
    ]
    WORDSIZE = len(sets[0])
    neighbors = [
        [
            i % 64
            for i in range(set_length * WORDSIZE)
            if st[-1 - i // 8] & (1 << (8 - 1 - i % 8))
        ]
        for st in sets
    ]
    return Graph(
        number_of_vertices=g.number_of_vertices,
        directed=g.directed,
        adjacency_dict={i: neighbors[i] for i in range(g.number_of_vertices)},
    )


def get_graph_from_atoms(atoms: ase.Atoms, only_top3=True) -> nx.Graph:
    """Get a networkx graph object from an ase.Atoms object.

    Args:
        atoms (ase.Atoms): The atoms object to convert to a graph.
        only_top3 (bool, optional): If True, only the top 3 shortest distances are used.
            Defaults to True, which is useful for fullerenes.

    Returns:
        nx.Graph: The networkx graph object.
    """
    cutoffs = natural_cutoffs(atoms)
    neighborList = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    adjacency_matrix = neighborList.get_connectivity_matrix(sparse=False)
    if only_top3:
        if (adjacency_matrix.sum(-1) != 3).any():
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            logger.warning(
                "More than 3 neighbors by ase found for input atoms"
                "Trying using first 3 shortest distance."
            )
            neighborDis: np.ndarray = atoms.get_all_distances()
            neighborSorted = np.argsort(neighborDis)
            adjacency_matrix[np.arange(len(atoms))[:, None], neighborSorted[:, 1:4]] = 1
    adjacency_matrix = np.array(adjacency_matrix)
    graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)
    for node, z in zip(graph.nodes, atoms.get_atomic_numbers()):
        graph.nodes[node]["z"] = z
    return graph


def nx_to_nauty(G, include_z_labels=True):
    """
    将 networkx 图转换为 pynauty 图
    """
    # 将 networkx 图转换为 pynauty 图
    import pynauty as pn

    adj_dict = {}
    for u, v in G.edges():
        adj_dict[u] = adj_dict.get(u, []) + [v]
    if include_z_labels:
        unique_elements, inverse_indices = np.unique(
            list(nx.get_node_attributes(G, "z").values()), return_inverse=True
        )
        index_dict = [
            set(np.nonzero(inverse_indices == i)[0])
            for i, _ in enumerate(unique_elements)
        ]
    nauty_graph = pn.Graph(
        number_of_vertices=G.number_of_nodes(),
        directed=False,
        adjacency_dict=adj_dict,
        vertex_coloring=index_dict if include_z_labels else [],
    )

    return nauty_graph
