import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from webscraping.core.models import CryptoData

# Encontrar a raiz do projeto (FinanceInsight)
BASE_DIR = Path(__file__).resolve().parents[4]  # Subindo quatro níveis agora

# Garantir que estamos na pasta correta
if not (BASE_DIR / "app").exists():
    raise RuntimeError(f"BASE_DIR incorreto: {BASE_DIR}")

print(f"BASE_DIR ajustado para: {BASE_DIR}")  # Depuração

# Definir diretório correto para os dados
DATA_DIR = BASE_DIR / "app" / "data_lake" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Criar o diretório, se necessário

def save_crypto_data(crypto_data: CryptoData):
    date_str = datetime.now().strftime("%Y-%m-%d")
    nome_arq_str = f"crypto_data_{date_str}"

    # Caminho para os arquivos
    file_path_xlsx = DATA_DIR / f"{nome_arq_str}.xlsx"
    file_path_parquet = DATA_DIR / f"{nome_arq_str}.parquet"

    print(f"Arquivos serão salvos em: {file_path_xlsx}")  # Depuração

    # Criar DataFrame
    df = pd.DataFrame([crypto_data.dict()])

    # Salvar em Excel
    if file_path_xlsx.exists():
        try:
            existing_df = pd.read_excel(file_path_xlsx, engine="openpyxl")
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Erro ao ler o arquivo Excel: {e}")
    df.to_excel(file_path_xlsx, index=False, engine="openpyxl")

    # Salvar em Parquet
    if file_path_parquet.exists():
        try:
            existing_df = pd.read_parquet(file_path_parquet)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Erro ao ler o arquivo Parquet: {e}")
    df.to_parquet(file_path_parquet, index=False, engine="pyarrow")
