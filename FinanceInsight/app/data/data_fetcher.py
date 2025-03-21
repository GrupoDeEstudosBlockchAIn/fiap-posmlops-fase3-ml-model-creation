import datetime
import pandas as pd
import yfinance as yf
import logging
from app.src.config import DATA_LAKE_RAW

# Configuração do log
logging.basicConfig(
    filename=DATA_LAKE_RAW / "fetch_crypto_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# Analisa as 100 criptomoedas
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
    """Coleta dados brutos do Yahoo Finance (via yfinance) e salva no diretório DATA_LAKE/RAW."""

    calculation_date = datetime.datetime.now().strftime("%Y-%m-%d")
    calculation_hour = datetime.datetime.now().strftime("%H:%M:%S")
    data = []
    
    for symbol in CRYPTO_SYMBOLS:
        try:
            crypto = yf.Ticker(symbol)
            info = crypto.info
            
            if not info:
                logging.warning(f"Nenhum dado disponível para {symbol}. Pulando...")
                continue
            
            data.append({
                "Symbol": symbol,
                "Name": info.get("shortName", symbol),
                "Price": info.get("regularMarketPrice", None),
                "Change": info.get("regularMarketChange", None),
                "Change %": info.get("regularMarketChangePercent", None),
                "Market Cap": info.get("marketCap", None),
                "Volume": info.get("regularMarketVolume", None),
                "Circulating Supply": info.get("circulatingSupply", None),
                "Calculation Date": calculation_date,
                "Calculation Hour": calculation_hour
            })
        
        except Exception as e:
            logging.error(f"Erro ao coletar dados de {symbol}: {e}")
            continue

    if not data:
        logging.error("Nenhum dado de mercado coletado. Encerrando processamento.")
        return None

    df = pd.DataFrame(data)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Salvando CSV
    raw_filename = DATA_LAKE_RAW / f"raw_{timestamp}.csv"
    df.to_csv(raw_filename, index=False, encoding="utf-8")

    # Salvando Excel
    excel_filename = DATA_LAKE_RAW / f"raw_{timestamp}.xlsx"
    df.to_excel(excel_filename, index=False)

    logging.info(f"Dados brutos salvos em: {raw_filename}")
    logging.info(f"Dados brutos salvos em: {excel_filename}")

    return raw_filename
