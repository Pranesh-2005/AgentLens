import os
import tempfile

import pytest

import agentlens
from agentlens import client as _client
from agentlens.server.db import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "agentlens.db"))


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Init AgentLens against a throwaway local DB and return the store."""
    db = str(tmp_path / "agentlens.db")
    _client._config.client = None
    agentlens.init(project="test", db_path=db)
    yield _client.get_client().store
    _client._config.client = None