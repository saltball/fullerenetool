from abc import ABC, abstractmethod
from typing import List

import ase
import networkx as nx


class BaseFullerene(ABC):
    """Base class for fullerene objects.

    The fullerene instance has the following properties and methods:
    - `atoms`: The ase.Atoms object representing the molecule.
    - `cage`: The CageGraph object representing the cage.
    - `__str__`: Brief information about the atoms and the cage.
    """

    def __init__(self, atoms: ase.Atoms):
        """Initializes a BaseFullerene object.

        Args:
            atoms (ase.Atoms): The atoms that make up the fullerene.
        """
        self._atoms = self._check_atoms(atoms)

    @property
    def atoms(self) -> ase.Atoms:
        """The ase.Atoms object representing the molecule."""
        return self._atoms

    @property
    @abstractmethod
    def graph(self) -> List[nx.Graph]:
        """The atomic connected components of the molecule."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cages(self) -> List:
        """The object representing the cage(s)."""
        return NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Brief information about the atoms and the cage."""
        raise NotImplementedError

    @abstractmethod
    def _check_atoms(self, atoms):
        raise NotImplementedError
