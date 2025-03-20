# Lista de criptomoedas (substitua ou expanda conforme necessário)
import datetime
import pandas as pd
import yfinance as yf
from app.src.config import DATA_LAKE_RAW


CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
    "BCH-USD", "LINK-USD", "XLM-USD", "UNI-USD", "ETC-USD", "FIL-USD", "ATOM-USD", "VET-USD", "HBAR-USD", "ICP-USD",
    "EGLD-USD", "MANA-USD", "SAND-USD", "AXS-USD", "THETA-USD", "AAVE-USD", "FTM-USD", "GRT-USD", "FLOW-USD", "KSM-USD",
    "XTZ-USD", "NEO-USD", "CHZ-USD", "CAKE-USD", "ZIL-USD", "ENJ-USD", "LRC-USD", "BAT-USD", "DASH-USD", "CRV-USD",
    "RUNE-USD", "1INCH-USD", "CELO-USD", "XEM-USD", "QTUM-USD", "ONT-USD", "OMG-USD", "YFI-USD", "ICX-USD", "BTT-USD",
    "WAVES-USD", "ZRX-USD", "KAVA-USD", "ANKR-USD", "ALGO-USD", "STX-USD", "DGB-USD", "HNT-USD", "KLAY-USD", "GALA-USD",
    "RSR-USD", "MINA-USD", "XDC-USD", "RVN-USD", "IOST-USD", "WAXP-USD", "SC-USD", "COTI-USD", "TEL-USD", "WIN-USD",
    "TWT-USD", "ZEN-USD", "AR-USD", "BNT-USD", "OGN-USD", "UMA-USD", "BAL-USD", "REEF-USD", "LPT-USD", "STORJ-USD",
    "RLC-USD", "NKN-USD", "OCEAN-USD", "BAND-USD", "SKL-USD", "MTL-USD", "CVC-USD", "PERP-USD", "STMX-USD", "CTSI-USD",
    "SXP-USD", "TRX-USD", "FTT-USD", "EWT-USD", "API3-USD", "MOVR-USD", "KMD-USD", "PUNDIX-USD", "GNO-USD", "MLN-USD"
]

def fetch_crypto_data():
    """Coleta dados brutos do Yahoo Finance (via yfinance) e salva no diretório RAW."""
    tickers = yf.Tickers(" ".join(CRYPTO_SYMBOLS))
    data = []
    
    calculation_date = datetime.datetime.now().strftime("%Y-%m-%d")
    calculation_hour = datetime.datetime.now().strftime("%H:%M:%S")

    for symbol in CRYPTO_SYMBOLS:
        crypto = tickers.tickers.get(symbol)
        if crypto:
            info = crypto.info
            data.append({
                "Symbol": symbol,
                "Name": info.get("shortName", symbol),
                "Price": info.get("regularMarketPrice"),
                "Change": info.get("regularMarketChange"),
                "Change %": info.get("regularMarketChangePercent"),
                "Market Cap": info.get("marketCap"),
                "Volume": info.get("regularMarketVolume"),
                "Circulating Supply": info.get("circulatingSupply"),
                "Calculation Date": calculation_date,
                "Calculation Hour": calculation_hour
            })

    if not data:
        print("⚠️ Nenhum dado de mercado coletado. Encerrando processamento.")
        return None

    df = pd.DataFrame(data)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Salvando CSV
    raw_filename = DATA_LAKE_RAW / f"raw_{timestamp}.csv"
    df.to_csv(raw_filename, index=False, encoding="utf-8")

    # Salvando Excel
    excel_filename = DATA_LAKE_RAW / f"raw_{timestamp}.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"✅ Dados brutos salvos em: {raw_filename}")
    print(f"✅ Dados brutos salvos em: {excel_filename}")

    return raw_filename