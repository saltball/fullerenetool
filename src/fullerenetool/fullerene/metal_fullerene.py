from abc import ABC, abstractmethod

import ase
import networkx as nx

from fullerenetool.fullerene import BaseFullerene, FullereneCage
from fullerenetool.operator.graph import get_graph_from_atoms


class MetalFullerene(BaseFullerene, ABC):
    @property
    @abstractmethod
    def metals(self):
        pass

    @property
    @abstractmethod
    def carbon_cages(self) -> FullereneCage:
        pass


class GeneralMetalFullerene(MetalFullerene):
    def __init__(self, atoms):
        self._atoms = atoms

    @property
    def carbon_cages(self):
        # we assume the largest connected carbon graph is the cage
        carbon_atoms = ase.Atoms(self.atoms[self.atoms.get_atomic_numbers() == 6])
        graph = get_graph_from_atoms(carbon_atoms, only_top3=False)
        subgraphs = list(sorted(nx.connected_components(graph)))
        return FullereneCage.from_atoms(ase.Atoms(carbon_atoms[list(subgraphs[0])]))
