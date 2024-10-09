try:
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
except ImportError:
    print(
        "Please install torch_dftd to use this calculator."
        " See https://github.com/pfnet-research/torch-dftd for more information.\n"
        "Or run `conda/mamba install torch-dftd -c xjtuicp` to install it."
    )

__all__ = ["TorchDFTD3Calculator"]
