import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

@pytest.fixture
def client():
    """Fixture that initializes the TestClient using a context manager to trigger lifespan events."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    """Tests the service status healthcheck root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "RAG Backend Scaffolding" in data["service"]

def test_ask_endpoint_success(client):
    """Tests the /ask endpoint returns a successful RAG response with mock data."""
    payload = {"query": "How does RAG function?", "k": 2}
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "How does RAG function?"
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert isinstance(data["context"], list)
    assert len(data["context"]) > 0
    assert "chunk_id" in data["context"][0]
    assert "content" in data["context"][0]
    assert data["is_grounded"] is True
    assert data["is_refused"] is False
    assert "elapsed_seconds" in data["metadata"]

def test_ask_endpoint_validation_invalid_k(client):
    """Tests that the /ask endpoint enforces boundary conditions on k."""
    # k must be between 1 and 50
    payload = {"query": "test query", "k": 100}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422  # Validation Error

def test_ask_endpoint_validation_empty_query(client):
    """Tests that the /ask endpoint enforces query string validation constraints."""
    payload = {"query": "", "k": 2}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422  # Validation Error
