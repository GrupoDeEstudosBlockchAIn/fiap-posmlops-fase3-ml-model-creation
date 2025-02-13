from pydantic import BaseModel
from typing import List, Optional

class CryptoData(BaseModel):
    name: str
    symbol: str
    market_cap: float
    price: float
    rank: int
    current_price: float
    price_change_24h: float
    price_change_percentage_24h: float
    total_volume: float
    market_cap_change_24h: float
    market_cap_change_percentage_24h: float
    circulating_supply: Optional[float]
    total_supply: Optional[float]
    ath: float
    ath_date: str
    atl: float
    atl_date: str
    high_24h: float
    low_24h: float
    last_updated: str
