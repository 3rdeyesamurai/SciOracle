import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="lexegis-tests-")
os.environ.setdefault("LEXEGIS_DATA_DIR", _TMP)
os.environ["LEXEGIS_DB"] = os.path.join(_TMP, "test.db")
os.environ["LEXEGIS_BLOBS"] = os.path.join(_TMP, "blobs")
os.environ["LEXEGIS_JWT_SECRET"] = "test-secret"

import pytest
from fastapi.testclient import TestClient

from app.db import init_db
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _db():
    init_db()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def account(client):
    import uuid
    email = f"counsel-{uuid.uuid4().hex[:8]}@example.com"
    response = client.post("/api/v1/auth/signup", json={
        "email": email, "password": "a-very-long-passphrase", "organisation": "Test Chambers"})
    assert response.status_code == 201, response.text
    body = response.json()
    return {"headers": {"Authorization": f"Bearer {body['access_token']}"}, **body, "email": email}
