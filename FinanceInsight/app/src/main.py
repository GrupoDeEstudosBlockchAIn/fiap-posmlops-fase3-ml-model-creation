import pandas as pd
import datetime
import time
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import inspect

# Encontrar a raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Definir diretórios corretos para os dados
RAW_DIR = BASE_DIR / "data_lake" / "raw"
REFINED_DIR = BASE_DIR / "data_lake" / "refined"
MODEL_DIR = BASE_DIR / "models"
DASH_DIR = BASE_DIR / "dashboards"
REPORT_DIR = BASE_DIR / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REFINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DASH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Lista de criptomoedas (substitua ou expanda conforme necessário)
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
    raw_filename = RAW_DIR / f"raw_{timestamp}.csv"
    df.to_csv(raw_filename, index=False, encoding="utf-8")

    # Salvando Excel
    excel_filename = RAW_DIR / f"raw_{timestamp}.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"✅ Dados brutos salvos em: {raw_filename}")
    print(f"✅ Dados brutos salvos em: {excel_filename}")

    return raw_filename

def process_crypto_data(raw_filename):
    """Processa os dados brutos e salva no diretório REFINED."""
    if raw_filename is None:
        print("⚠️ Nenhum arquivo bruto disponível. Pulando processamento.")
        return

    df = pd.read_csv(raw_filename)

    if df.empty:
        print("⚠️ Arquivo CSV está vazio. Encerrando processamento.")
        return

    df.dropna(inplace=True)
    
    for col in ["Price", "Change", "Change %", "Market Cap", "Volume", "Circulating Supply"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(inplace=True)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    refined_filename = REFINED_DIR / f"refined_{timestamp}.parquet"
    df.to_parquet(refined_filename, index=False)

    print(f"✅ Dados refinados salvos em: {refined_filename}")

# Função para treinar e prever com o modelo de Machine Learning
def train_and_predict():
    """Treina o modelo e faz previsões para as 10 criptomoedas favoritas."""
    files = list(REFINED_DIR.glob("*.parquet"))
    if not files:
        print("⚠️ Nenhum dado refinado disponível.")
        return
    
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    df["Calculation Date"] = pd.to_datetime(df["Calculation Date"])
    df.sort_values("Calculation Date", inplace=True)
    
    features = ["Market Cap", "Volume"]
    target = "Price"
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    
    mape = (abs(y_test - y_pred) / y_test).mean() * 100
    
    # Cálculo da "acurácia" para regressão: % de previsões dentro de um erro aceitável
    tolerance = 0.05  # 5% de tolerância
    accuracy = sum(abs(y_test - y_pred) / y_test < tolerance) / len(y_test) * 100
    
    # F1 Score para problemas de regressão não é convencional, então aqui está apenas um placeholder
    f1 = f1_score((y_test > y_test.median()).astype(int), (y_pred > y_test.median()).astype(int))
    
    print(f"📊 MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Acurácia: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = model.predict(df[features].iloc[-10:])
    
    # RELATÓRIO DE MÉTRICAS DE DESEMPENHO DE MODELOS PREDITIVOS    
    predictive_model_performance_metrics_report(mae, mse, rmse, mape, accuracy, f1, df_predictions[["Symbol", "Predicted Price"]])
    
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo em: {model_path}")
    
    print("🔮 Top 10 previsões de preços:")
    print(df_predictions[["Symbol", "Predicted Price"]])


# Função para gerar relatório de métricas
def predictive_model_performance_metrics_report(mae, mse, rmse, mape, accuracy, f1, df_predictions):
    """Gera um relatório HTML com as métricas de desempenho do modelo preditivo."""
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    report_filename = REPORT_DIR / f"report_{timestamp}.html"
    
    # Renomeia colunas para português
    df_predictions = df_predictions.rename(columns={"Symbol": "Símbolo", "Predicted Price": "Preço Previsto"})
    
    report_content = f"""
    <html>
    <head>
        <title>Relatório de Métricas do Modelo</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #2c3e50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Métricas do Modelo Preditivo</h1>
        <h2>Métricas de Desempenho</h2>
        <table>
            <tr><th>Métrica</th><th>Valor</th></tr>
            <tr><td>Erro Médio Absoluto (MAE)</td><td>{mae:.4f}</td></tr>
            <tr><td>Erro Quadrático Médio (MSE)</td><td>{mse:.4f}</td></tr>
            <tr><td>Raiz do Erro Quadrático Médio (RMSE)</td><td>{rmse:.4f}</td></tr>
            <tr><td>Erro Percentual Absoluto Médio (MAPE)</td><td>{mape:.2f}%</td></tr>
            <tr><td>Acurácia</td><td>{accuracy:.2f}%</td></tr>
            <tr><td>F1 Score</td><td>{f1:.4f}</td></tr>
        </table>
        
        <h2>Previsões das Top 10 Criptomoedas</h2>
        {df_predictions.to_html(index=False)}
    </body>
    </html>
    """
    
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(report_content)
    
    print(f"✅ Relatório salvo em: {report_filename}")



def dashboards_cryptocurrency_forecast():
    """Gera e salva dashboards das previsões de preços das 10 criptomoedas favoritas."""
    files = list(REFINED_DIR.glob("*.parquet"))
    if not files:
        print("⚠️ Nenhum dado refinado disponível para gerar dashboards.")
        return
    
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df["Calculation Date"] = pd.to_datetime(df["Calculation Date"])
    df.sort_values("Calculation Date", inplace=True)
    
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    if not model_path.exists():
        print("⚠️ Modelo de previsão não encontrado.")
        return
    
    model = joblib.load(model_path)
    features = ["Market Cap", "Volume"]
    latest_data = df[features].iloc[-10:]
    predictions = model.predict(latest_data)
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = predictions
    
    # Criando cores aleatórias para cada criptomoeda
    import random
    colors = [f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 0.8)' for _ in range(len(df_predictions))]
    
    # Criando o gráfico de barras com tooltip customizado
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_predictions["Name"],  # Agora exibe o nome da criptomoeda em vez do símbolo
        y=df_predictions["Predicted Price"],
        text=None,  # Removendo qualquer texto dentro das barras
        marker=dict(color=colors),
        hovertext=[f"{symbol}, ${price:.2f}"  # Adicionando o cifrão ($) no preço previsto
                   for symbol, price in zip(df_predictions["Symbol"], df_predictions["Predicted Price"])],
        hoverinfo="text"  # Define que o tooltip exibe apenas o texto personalizado
    ))
    
    fig.update_layout(
        title="Previsão de Preços das Top 10 Criptomoedas",
        xaxis_title="Criptomoedas",
        yaxis_title="Preço Previsto (USD)",
        template="plotly_dark"
    )
    
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dash_filename = DASH_DIR / f"dash_{timestamp}.html"
    fig.write_html(dash_filename)
    
    print(f"✅ Dashboard salvo em: {dash_filename}")


def main():
    while True:
        # Faz o scraping e obtém o dados brutos
        raw_file = fetch_crypto_data()

        # Faz o refinamento dos dados brutos
        process_crypto_data(raw_file)

        # Faz o treinamento do modelo de ML de Previsão
        train_and_predict()

        # Dashboards de Previsão de Criptomoedas
        dashboards_cryptocurrency_forecast()

        print("⏳ Aguardando 1 hora para a próxima execução...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
