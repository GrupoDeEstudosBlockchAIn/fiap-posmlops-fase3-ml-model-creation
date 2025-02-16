import pytest
from httpx import AsyncClient, ASGITransport
from app.src.main import app  # Certifique-se de que está importando corretamente

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as ac:
        response = await ac.get("/", follow_redirects=True)  # Segue o redirecionamento    
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Confirma que a resposta é uma lista
    assert len(response.json()) > 0  # Verifica se há criptomoedas retornadas

@pytest.mark.asyncio
async def test_docs():
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as ac:
        response = await ac.get("/docs")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_healthcheck():
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as ac:
        response = await ac.get("/health")
    
    if response.status_code == 404:
        pytest.fail("A rota /health não está registrada. Verifique se foi adicionada corretamente no FastAPI.")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
