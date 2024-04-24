import ase
import numpy as np


class InvalidMoleculeError(ValueError):
    """The molecule is invalid."""

    pass


class AtomTooCloseError(InvalidMoleculeError):
    """The atoms are too close to each other."""

    pass


def valid_molecule(
    atoms: ase.Atoms, distance_threshold=0.5, *, skip_ghost=True, raise_error=True
):
    """Check if the molecule is valid.

    This function checks if the given molecule is valid by performing the following
    checks:
    1. If skip_ghost is True, it filters out any ghost atoms from the given atoms.
    2. It checks if any atoms are too close to each other based on the
        distance_threshold.

    Args:
        atoms (ase.Atoms): The atoms to check.
        distance_threshold (float, optional): The distance threshold to check if the
            atoms are too close to each other.
        skip_ghost (bool): If True, ghost atoms will be skipped during the check.
        raise_error (bool): If True, an error will be raised if the molecule is invalid.

    Returns:
        None

    Raises:
        ValueError: If raise_error is True and the molecule is invalid.
    """
    if skip_ghost:  # We skip check ghost atoms
        atoms = filter_ghost_atom(atoms)
    return _check_too_close(atoms, distance_threshold, raise_error=raise_error)


def _check_too_close(
    atoms: ase.Atoms,
    threshold: float = 0.5,
    *,
    raise_error=True,
) -> bool:
    """Check if the atoms are too close to each other.

    Args:
        atoms (ase.Atoms): The atoms to check.
        threshold (float): The distance threshold to check if the atoms are too close
            to each other.
        raise_error (bool): Flag indicating whether to raise an error if atoms are too
            close. If False, return False instead of raising error.

    Raises:
        AtomTooCloseError: If atoms are too close to each other.

    """
    if (atoms.get_all_distances() + np.eye(len(atoms)) < threshold).any():
        first_pair: ase.Atoms = ase.Atoms(
            atoms[np.argwhere(atoms.get_all_distances() < threshold)[1]]
        )
        if raise_error:
            raise AtomTooCloseError(
                "Atoms are too close to each other: {}".format(
                    ", ".join(
                        "{}({})".format(atom.symbol, list(atom.position))
                        for atom in first_pair
                    )
                )
            )
        return False
    return True


def filter_ghost_atom(atoms, *, ghost_atomic_nubmer=None):
    if ghost_atomic_nubmer is None:
        ghost_atomic_nubmer = 0
    return ase.Atoms(atoms[atoms.get_atomic_numbers() != ghost_atomic_nubmer])
