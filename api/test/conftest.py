import pytest
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'api' package is importable when pytest
# adjusts working directories during collection.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()
