import random

import dgl
import numpy as np
import pytest
import backend as F


SEED = 12345


# Before each test, seed the random number generator. This makes explicit
# randomness in tests fully reproducible regardless of which tests are run or in
# which order.
@pytest.fixture(autouse=True)
def seed_random():
    random.seed(SEED)  # For networkx
    np.random.seed(SEED)  # For scipy
    dgl.seed(SEED)
    F.seed(SEED)
