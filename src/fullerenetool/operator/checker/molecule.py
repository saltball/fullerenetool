import ase
import numpy as np


class InvalidMoleculeError(ValueError):
    """The molecule is invalid."""

    pass


class AtomTooCloseError(InvalidMoleculeError):
    """The atoms are too close to each other."""

    pass


def valid_molecule(atoms: ase.Atoms, distance_threshold=0.5):
    """Check if the molecule is valid.

    Args:
        atoms (ase.Atoms): The atoms to check.
        distance_threshold (float, optional): The distance threshold to check if the
            atoms are too close to each other.
    """
    _check_too_close(atoms, distance_threshold)


def _check_too_close(atoms: ase.Atoms, threshold: float = 0.5):
    """Check if the atoms are too close to each other.

    Args:
        atoms (ase.Atoms): The atoms to check.
        threshold (float): The distance threshold to check if the atoms are too close
            to each other.

    Raises:
        AtomTooCloseError: _description_
    """
    if (atoms.get_all_distances() + np.eye(len(atoms)) < threshold).any():
        first_pair: ase.Atoms = ase.Atoms(
            atoms[np.argwhere(atoms.get_all_distances() < threshold)[0]]
        )
        raise AtomTooCloseError(
            "Atoms are too close to each other: {}".format(
                ", ".join(
                    "{}({})".format(atom.symbol, list(atom.position))
                    for atom in first_pair
                )
            )
        )
