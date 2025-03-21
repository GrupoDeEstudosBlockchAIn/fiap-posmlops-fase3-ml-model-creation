import datetime
import pandas as pd
from app.data.feature_engineering import calculate_bollinger_bands, calculate_macd, calculate_rsi
from app.src.config import DATA_LAKE_REFINED

def process_crypto_data(raw_filename):
    """Processa os dados brutos e salva no diretório REFINED com novas features técnicas."""
    if raw_filename is None:
        print("Nenhum arquivo bruto disponível. Pulando processamento.")
        return

    df = pd.read_csv(raw_filename)

    if df.empty:
        print("Arquivo CSV está vazio. Encerrando processamento.")
        return

    df.dropna(inplace=True)

    for col in ["Price", "Change", "Change %", "Market Cap", "Volume", "Circulating Supply"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Adicionando novas médias móveis
    df["SMA_3"] = df["Price"].rolling(window=3).mean()
    df["SMA_7"] = df["Price"].rolling(window=7).mean()
    df["SMA_14"] = df["Price"].rolling(window=14).mean()

    # Cálculo do RSI
    df["RSI_14"] = calculate_rsi(df["Price"], period=14)

    # Cálculo do MACD
    df["MACD"], df["MACD_Signal"] = calculate_macd(df["Price"])

    # Cálculo das Bandas de Bollinger
    df["BB_Mean"], df["BB_Upper"], df["BB_Lower"] = calculate_bollinger_bands(df["Price"], window=20)
    
    # Salvando os dados refinados
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    refined_filename = DATA_LAKE_REFINED / f"refined_{timestamp}.parquet"
    df.to_parquet(refined_filename, index=False)

    print(f"Dados refinados salvos em: {refined_filename}")
