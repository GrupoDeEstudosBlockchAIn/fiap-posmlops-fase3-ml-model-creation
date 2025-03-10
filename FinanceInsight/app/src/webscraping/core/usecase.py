from typing import List
from app.src.webscraping.infrastructure.cripto_scraper import fetch_top_cryptos
from app.src.webscraping.infrastructure.storage import save_crypto_data
from app.src.webscraping.core.models import CryptoData

def get_and_store_top_cryptos(limit: int = 100) -> List[CryptoData]:
    cryptos = fetch_top_cryptos(limit)
    for crypto in cryptos:
        save_crypto_data(crypto)
    return cryptos
