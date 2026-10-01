import pytest

from ceb.config import load_params, smoke_params


@pytest.fixture(scope="session")
def P():
    return load_params()


@pytest.fixture(scope="session")
def SP():
    return smoke_params()
