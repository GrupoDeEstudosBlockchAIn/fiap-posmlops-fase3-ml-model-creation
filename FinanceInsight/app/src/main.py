import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    """Processa os dados brutos e salva no diretório REFINED com novas features técnicas."""
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

    # Adicionando novas features técnicas
    df["SMA_7"] = df["Price"].rolling(window=7).mean()
    df["SMA_30"] = df["Price"].rolling(window=30).mean()
    df["EMA_7"] = df["Price"].ewm(span=7, adjust=False).mean()
    df["Volatilidade_7"] = df["Price"].rolling(window=7).std()
      
    # Salvando os dados refinados
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    refined_filename = REFINED_DIR / f"refined_{timestamp}.parquet"
    df.to_parquet(refined_filename, index=False)

    print(f"✅ Dados refinados salvos em: {refined_filename}")



# Função para remover outliers usando o método IQR
def remove_outliers(df, columns):
    """Remove outliers das colunas especificadas usando o método do Intervalo Interquartil (IQR)."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


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
    
    features = ["Market Cap", "Volume", "SMA_7", "SMA_30", "EMA_7", "Volatilidade_7"]
    target = "Price"
    
    # Removendo outliers
    df = remove_outliers(df, features + [target])
    
    X = df[features]
    y = df[target]
    
    # Normalizando os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Converter X_train e X_test para DataFrames com os mesmos nomes de colunas
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    
    # Reduzindo n_estimators para evitar overfitting
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    smape = (2 * abs(y_test - y_pred) / (abs(y_test) + abs(y_pred))).mean() * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%, R²: {r2:.4f}")
    
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = model.predict(pd.DataFrame(scaler.transform(df[features].iloc[-10:]), columns=features))
    
    # Verificando a importância das features
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("📌 Importância das Features:")
    print(feature_importances)
    
    # Relatório de métricas
    predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_predictions[["Symbol", "Predicted Price", "Name"]])
    
    # Salvando modelo e scaler
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Modelo salvo em: {model_path}")
    print(f"✅ Scaler salvo em: {scaler_path}")
    
    print("🔮 Top 10 previsões de preços:")
    print(df_predictions[["Symbol", "Predicted Price"]])

def predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_predictions):
    """Gera um relatório HTML com as métricas de desempenho do modelo preditivo e chama o dashboard."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    report_filename = REPORT_DIR / f"report_{timestamp}.html"
    
    # Renomeia colunas e formata valores
    df_predictions = df_predictions.rename(columns={"Symbol": "Símbolo", "Predicted Price": "Preço Previsto", "Name": "Nome"})
    df_predictions = df_predictions[["Nome", "Símbolo", "Preço Previsto"]]
    df_predictions["Preço Previsto"] = df_predictions["Preço Previsto"].apply(lambda x: f"${x:.2f}")
    
    # Chama a função do dashboard passando os dados processados
    dash_filename = dashboards_cryptocurrency_forecast(df_predictions)
    
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
        <p><strong>Data da Previsão:</strong> {now.strftime('%d/%m/%Y')}</p>
        <p><strong>Horário da Previsão:</strong> {now.strftime('%H:%M')}</p>
        
        <h2>Previsões das Top 10 Criptomoedas</h2>
        {df_predictions.to_html(index=False)}
        
        <h2>Métricas de Desempenho</h2>
        <table>
            <tr><th>Métrica</th><th>Valor</th></tr>
            <tr><td>Erro Médio Absoluto (MAE)</td><td>{mae:.2f}</td></tr>
            <tr><td>Erro Quadrático Médio (MSE)</td><td>{mse:.2f}</td></tr>
            <tr><td>Raiz do Erro Quadrático Médio (RMSE)</td><td>{rmse:.2f}</td></tr>
            <tr><td>Erro Percentual Absoluto Médio Simétrico (SMAPE)</td><td>{smape:.2f}%</td></tr>
            <tr><td>Coeficiente de Determinação (R²)</td><td>{r2:.2f}</td></tr>
        </table>
        
        <h2>Dashboard</h2>
        <p><a href="{dash_filename}" target="_blank">Clique aqui para visualizar o Dashboard</a></p>
    </body>
    </html>
    """
    
    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(report_content)
    
    print(f"✅ Relatório salvo em: {report_filename}")


def dashboards_cryptocurrency_forecast(df_predictions):
    """Gera e salva dashboards das previsões recebidas."""
    now = datetime.datetime.now()
    forecast_date = now.strftime('%d/%m/%Y')
    forecast_time = now.strftime('%H:%M')
    
    import random
    colors = [f'rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, 0.8)' for _ in range(len(df_predictions))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_predictions["Nome"],
        y=df_predictions["Preço Previsto"].str.replace("$", "").astype(float),
        marker=dict(color=colors),
        hovertext=[f"{symbol}, {price}" for symbol, price in zip(df_predictions["Símbolo"], df_predictions["Preço Previsto"])],
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=f"Previsão de Preços das Top 10 Criptomoedas ({forecast_date} - {forecast_time})",
        xaxis_title="Criptomoedas",
        yaxis_title="Preço Previsto (USD)",
        template="plotly_dark"
    )
    
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    dash_filename = DASH_DIR / f"dash_{timestamp}.html"
    fig.write_html(dash_filename)
    
    print(f"✅ Dashboard salvo em: {dash_filename}")
    return dash_filename


# def debug_mape(y_real, y_pred):
#     """Depura o cálculo do MAPE, identificando possíveis problemas"""
    
#     # Convertendo para arrays numpy
#     y_real = np.array(y_real)
#     y_pred = np.array(y_pred)

#     # Calcula o MAPE original
#     mape_original = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

#     # Identifica valores problemáticos (y_real próximo de zero)
#     valores_zeros = y_real < 0.01
#     if np.any(valores_zeros):
#         print(f"⚠️ ALERTA: {np.sum(valores_zeros)} valores reais são menores que 0.01, causando instabilidade no MAPE.")

#     # Calcula o MAPE ajustado (ignorando valores muito pequenos no denominador)
#     y_real_ajustado = np.where(y_real < 0.01, np.nan, y_real)  # Substitui valores muito pequenos por NaN
#     mape_ajustado = np.nanmean(np.abs((y_real - y_pred) / y_real_ajustado)) * 100  # Ignora NaNs

#     # Mostra os valores reais e previstos para análise
#     print("\nAmostra dos valores reais e previstos:")
#     for real, pred in zip(y_real[:10], y_pred[:10]):  # Exibe os primeiros 10 valores
#         print(f"Real: {real:.5f} | Previsto: {pred:.5f}")

#     # Exibe os valores do MAPE
#     print(f"\n📌 MAPE Original: {mape_original:.2f}%")
#     print(f"✅ MAPE Ajustado (Ignorando valores < 0.01): {mape_ajustado:.2f}%")

#     # Gráfico para visualização dos valores reais vs previstos
#     plt.figure(figsize=(8, 5))
#     plt.scatter(y_real, y_pred, alpha=0.6, color='blue', label="Valores")
#     plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], linestyle='dashed', color='red', label="Ideal (y=x)")
#     plt.xlabel("Valor Real")
#     plt.ylabel("Valor Previsto")
#     plt.title("Debug MAPE: Valores Reais vs Previstos")
#     plt.legend()
#     plt.grid()
#     plt.show()


def main():
    while True:
        # Faz o scraping e obtém o dados brutos
        raw_file = fetch_crypto_data()

        # Faz o refinamento dos dados brutos
        process_crypto_data(raw_file)

        # Faz o treinamento do modelo de ML de Previsão
        train_and_predict()

        print("⏳ Aguardando 1 hora para a próxima execução...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
