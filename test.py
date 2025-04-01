import sys

import pytest

from ketchup.data import initialize_data

from tests.utils import db_name


if __name__ == '__main__':
    initialize_data(dummy=True, db_name=db_name, batch_size=128)
    sys.exit(pytest.main(['tests/*.py']))
