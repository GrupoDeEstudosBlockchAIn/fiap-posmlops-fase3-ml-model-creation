from fastapi import APIRouter
from app.src.webscraping.core.usecase import get_and_store_top_cryptos

router = APIRouter()

@router.get("/cryptos", summary="Obter as 100 principais criptomoedas")
def get_top_cryptos(limit: int = 100):
    return get_and_store_top_cryptos(limit)
