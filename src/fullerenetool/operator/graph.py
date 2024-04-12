import ase
import networkx as nx
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from fullerenetool.logger import logger


def get_graph_from_atoms(atoms: ase.Atoms, only_top3=True) -> nx.Graph:
    """Get a networkx graph object from an ase.Atoms object.

    Args:
        atoms (ase.Atoms): The atoms object to convert to a graph.
        only_top3 (bool, optional): If True, only the top 3 shortest distances are used.
            Defaults to True for fullerene cages.

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
    return nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)
