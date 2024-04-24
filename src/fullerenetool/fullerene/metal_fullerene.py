from abc import ABC, abstractmethod
from typing import List

import ase
import networkx as nx

from fullerenetool.fullerene import BaseFullerene, FullereneCage
from fullerenetool.operator.graph import get_graph_from_atoms


class BaseMetalFullerene(BaseFullerene, ABC):
    @property
    @abstractmethod
    def metal_clusters(self) -> List[ase.Atoms]:
        """Metal atoms of the MetalFullerene

        Returns:
            ase.Atoms: metal atoms, connected in nature cut off from the fullerene
        """
        pass

    @property
    @abstractmethod
    def cages(self) -> List[FullereneCage]:
        pass


class MetalSingleFullerene(BaseMetalFullerene):
    def __init__(self, atoms):
        self._atoms = atoms

    @property
    def cage(self):
        # we assume the largest connected carbon graph is the cage
        carbon_atoms = ase.Atoms(self.atoms[self.atoms.get_atomic_numbers() == 6])
        graph = get_graph_from_atoms(carbon_atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        return FullereneCage.from_atoms(ase.Atoms(carbon_atoms[list(subgraphs[0])]))

    @property
    def cages(self):
        return [self.cage]

    @property
    def metal_clusters(self):
        return [ase.Atoms(self.atoms[self.atoms.get_atomic_numbers() != 6])]
