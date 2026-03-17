import grums
from grums.utils import set_global_seed


def test_package_import_and_version() -> None:
    assert grums.__version__ == "0.1.0"


def test_seed_utility_runs() -> None:
    set_global_seed(42)
