try:
    import torch  # noqa: F401
except ImportError:
    print(
        "Warning: PyTorch not installed. Mace needs PyTorch to run."
        " Please install PyTorch to use this calculator. See"
        " https://pytorch.org/ for more information.\n"
        "Or you can"
        " run `mamba/conda install pytorch` in your conda environment."
        " run `mamba/conda install pytorch pytorch-cuda` if you have NVidia GPU."
    )
try:
    from mace.calculators import mace_anicc, mace_mp, mace_off
except ImportError:
    print(
        "Warning: MACE not installed. Mace needs MACE to run. Please"
        " install MACE to use this calculator. See"
        " https://mace-docs.readthedocs.io/en/latest/guide/installation.html for more"
        " information. \n"
        "Or you can run `mamba/conda install pymace` in your conda environment."
    )

__all__ = ["mace_mp", "mace_off", "mace_anicc"]
