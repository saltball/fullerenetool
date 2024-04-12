from abc import ABC, abstractmethod
from functools import cached_property

import ase
import networkx as nx
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from fullerenetool.logger import logger


class BaseFullerene(ABC):
    @abstractmethod
    def _init_check(self):
        pass

    @property
    @abstractmethod
    def atoms(self) -> ase.Atoms:
        pass


class FullereneCage(BaseFullerene):
    """Initializes a FullereneCage object.

    The `FullereneCage` is expected to be a full-carbon cage with no
    restrictions on the number of atoms or the shape of the cage.

    Args:
        atoms (ase.Atoms): The atoms that make up the fullerene.
    """

    def __init__(self, atoms: ase.Atoms):
        self._atoms = atoms
        self._init_check()

    def _init_check(self):
        if not isinstance(self.atoms, ase.Atoms):
            raise ValueError("The atoms object must be an instance of ase.Atoms.")
        if not self.atoms:
            raise ValueError("The atoms object must be provided")
        if not (self.atoms.get_atomic_numbers() == 6).all():
            raise ValueError(
                "The atoms object must be made up of only carbon atoms, "
                "got compound: {}".format(self.atoms.symbols)
            )
        if len(self.atoms) < 1:
            raise ValueError("The atoms object must have at least one atom.")

    @property
    def atoms(self):
        return self._atoms

    @property
    def natoms(self) -> int:
        """The number of atoms in the fullerene."""
        return len(self.atoms)

    @cached_property
    def fullerenegraph(self) -> "FullereneGraph":
        """The fullerene graph object of the fullerene."""
        return self._get_graph()

    def _get_graph(self) -> "FullereneGraph":
        return FullereneGraph.from_fullerene(self)

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms):
        """Creates a FullereneCage object from an ase.Atoms object."""
        if not isinstance(atoms, ase.Atoms):
            raise ValueError(
                "The atoms object must be an instance of ase.Atoms, got"
                + str(type(atoms))
            )
        return cls(atoms)


class FullereneGraph(object):
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
    ):
        """Initializes a FullereneGraph object.

        The FullereneGraph object is a representation of a fullerene graph. It
        assumes that the fullerene is a carbon cage with no restrictions on the
        number of atoms or the shape of the cage.

        All carbon atoms are connected to each other by bonds.

        Note the following for inputs:
        - The adjacency matrix is a square matrix that represents the bonds
          between atoms. The matrix is symmetric and the diagonal is zero.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix of the cage.
        """
        if adjacency_matrix is None:
            raise ValueError(
                "One of the adjacency_matrix or circle_matrix is required."
            )
        self.adjacency_matrix = adjacency_matrix
        self.circle_matrix = self._get_circle_matrix()
        self._init_check()

    @cached_property
    def graph(self) -> nx.Graph:
        """The networkx graph object of the fullerene graph."""
        return nx.from_numpy_array(self.adjacency_matrix, create_using=nx.Graph)

    def _get_circle_matrix(self):
        from fullerenetool.algorithm import dual_graph

        circle_finder = dual_graph.py_graph_circle_finder(
            int(len(self.graph.edges)), np.array(self.graph.edges).astype(int).data
        )
        edge_list = circle_finder.get_dual_edge_list()
        circleADJ = np.zeros([circle_finder.face_size, circle_finder.face_size])
        for item in edge_list:
            circleADJ[item[0], item[1]] = 1
            circleADJ[item[1], item[0]] = 1
        return circleADJ

    def _init_check(self):
        # check the graph is connected all, not seperate
        if not nx.is_connected(self.graph):
            subgraphs = [
                len(c)
                for c in sorted(
                    nx.connected_components(self.graph), key=len, reverse=True
                )
            ]
            raise ValueError(
                "The graph must be connected, got {} graph componets: {}".format(
                    len(subgraphs), subgraphs
                )
            )

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms):
        """Creates a FullereneGraph object from an ase.Atoms object."""
        fullerene = FullereneCage(atoms)
        return cls.from_fullerene(fullerene)

    @classmethod
    def from_fullerene(cls, fullerene: FullereneCage):
        """Creates a FullereneGraph object from a FullereneCage object."""
        if not isinstance(fullerene, FullereneCage):
            raise ValueError(
                "The fullerene object must be an instance of GeneralFullerene."
            )
        fullerene_atoms = fullerene.atoms
        cutoffs = natural_cutoffs(fullerene_atoms)
        neighborList = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighborList.update(fullerene_atoms)
        adjacency_matrix = neighborList.get_connectivity_matrix(sparse=False)
        if (adjacency_matrix.sum(-1) != 3).any():
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            logger.warning(
                "More than 3 neighbors by ase found for input atoms"
                "Trying using first 3 shortest distance."
            )
            neighborDis: np.ndarray = fullerene_atoms.get_all_distances()
            neighborSorted = np.argsort(neighborDis)
            adjacency_matrix[
                np.arange(fullerene.natoms)[:, None], neighborSorted[:, 1:4]
            ] = 1
        adjacency_matrix = np.array(adjacency_matrix)
        return FullereneGraph(adjacency_matrix=adjacency_matrix)

    def __str__(self) -> str:
        return "FullereneGraph object with {} nodes and {} edges".format(
            self.graph.number_of_nodes(), self.graph.number_of_edges()
        )
