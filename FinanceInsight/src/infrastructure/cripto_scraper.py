import requests
from fastapi import HTTPException
from src.core.models import CryptoData

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_top_cryptos(limit: int = 100):
    try:
        response = requests.get(f"{COINGECKO_API}/coins/markets", params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1
        })
        response.raise_for_status()
        data = response.json()

        cryptos = [
            CryptoData(
                name=item["name"],
                symbol=item["symbol"],
                market_cap=item["market_cap"],
                price=item["current_price"],
                rank=item["market_cap_rank"],
                current_price=item["current_price"],
                price_change_24h=item["price_change_24h"],
                price_change_percentage_24h=item["price_change_percentage_24h"],
                total_volume=item["total_volume"],
                market_cap_change_24h=item["market_cap_change_24h"],
                market_cap_change_percentage_24h=item["market_cap_change_percentage_24h"],
                circulating_supply=item["circulating_supply"],
                total_supply=item["total_supply"],
                ath=item["ath"],
                ath_date=item["ath_date"],
                atl=item["atl"],
                atl_date=item["atl_date"],
                high_24h=item["high_24h"],
                low_24h=item["low_24h"],
                last_updated=item["last_updated"]

            )
            for item in data
        ]
        return cryptos
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao buscar criptomoedas: {e}")
