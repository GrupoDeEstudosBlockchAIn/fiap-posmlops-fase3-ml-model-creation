import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from scipy.stats import zscore
# import lightgbm as lgb
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam

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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

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
    
    # Definição do modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Espaço de busca dos hiperparâmetros
    param_dist = {
        "n_estimators": [50, 100, 200, 500],  # Número de árvores
        "max_depth": [3, 5, 7, 10],  # Profundidade máxima
        "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Taxa de aprendizado
        "subsample": [0.6, 0.8, 1.0],  # Amostragem de dados para evitar overfitting
        "colsample_bytree": [0.6, 0.8, 1.0],  # Amostragem de colunas para cada árvore
        "gamma": [0, 0.1, 0.2, 0.3],  # Redução mínima na perda para dividir um nó
        "reg_lambda": [0.01, 0.1, 1, 10],  # Regularização L2
        "reg_alpha": [0.01, 0.1, 1, 10]  # Regularização L1
    }

    # RandomizedSearchCV para encontrar os melhores hiperparâmetros
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, 
        n_iter=20, cv=3, scoring="neg_mean_absolute_error", 
        n_jobs=-1, verbose=1, random_state=42
    )

    # Ajustando o modelo com os melhores hiperparâmetros
    random_search.fit(X_train, y_train)

    # Obtendo os melhores hiperparâmetros encontrados
    best_params = random_search.best_params_
    print("🎯 Melhores Hiperparâmetros Encontrados:", best_params)

    # Treinar modelo final com os melhores hiperparâmetros
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    smape = (2 * abs(y_test - y_pred) / (abs(y_test) + abs(y_pred))).mean() * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%, R²: {r2:.4f}")
    
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = best_model.predict(pd.DataFrame(scaler.transform(df[features].iloc[-10:]), columns=features))
    
    # Verificando a importância das features
    feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print("📌 Importância das Features:")
    print(feature_importances)
    
    # Relatório de métricas
    predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_predictions[["Symbol", "Predicted Price", "Name"]])
    
    # Salvando modelo e scaler
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(best_model, model_path)
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

    # 🔹 Valores médios esperados para comparação + Tooltips
    expected_values = {
        "MAE": (0.5, "Quanto menor, melhor"),
        "MSE": (0.3, "Quanto menor, melhor"),
        "RMSE": (0.5, "Quanto menor, melhor"),
        "SMAPE": (10.0, "Ideal abaixo de 10%"),
        "R²": (0.85, "Quanto mais próximo de 1, melhor")
    }

    # 🔹 Tooltips para as métricas
    metric_tooltips = {
        "MAE": "Erro médio absoluto - Mede a média dos erros absolutos das previsões.",
        "MSE": "Erro quadrático médio - Penaliza mais os erros grandes devido à elevação ao quadrado.",
        "RMSE": "Raiz do erro quadrático médio - Dá mais peso a erros grandes e mantém a unidade original.",
        "SMAPE": "Erro percentual absoluto médio simétrico - Mede a precisão da previsão em porcentagem.",
        "R²": "Coeficiente de determinação - Mede o quão bem o modelo explica os dados. Quanto mais próximo de 1, melhor."
    }

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
            .tooltip {{
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted black;
                cursor: help;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 220px;
                background-color: black;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 100%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
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
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
                <th>Valor Médio Esperado</th>
            </tr>
            <tr>
                <td>
                    <div class="tooltip">Erro Médio Absoluto (MAE)
                        <span class="tooltiptext">{metric_tooltips["MAE"]}</span>
                    </div>
                </td>
                <td>{mae:.2f}</td>
                <td>
                    <div class="tooltip">{expected_values["MAE"][0]:.2f}
                        <span class="tooltiptext">{expected_values["MAE"][1]}</span>
                    </div>
                </td>
            </tr>
            <tr>
                <td>
                    <div class="tooltip">Erro Quadrático Médio (MSE)
                        <span class="tooltiptext">{metric_tooltips["MSE"]}</span>
                    </div>
                </td>
                <td>{mse:.2f}</td>
                <td>
                    <div class="tooltip">{expected_values["MSE"][0]:.2f}
                        <span class="tooltiptext">{expected_values["MSE"][1]}</span>
                    </div>
                </td>
            </tr>
            <tr>
                <td>
                    <div class="tooltip">Raiz do Erro Quadrático Médio (RMSE)
                        <span class="tooltiptext">{metric_tooltips["RMSE"]}</span>
                    </div>
                </td>
                <td>{rmse:.2f}</td>
                <td>
                    <div class="tooltip">{expected_values["RMSE"][0]:.2f}
                        <span class="tooltiptext">{expected_values["RMSE"][1]}</span>
                    </div>
                </td>
            </tr>
            <tr>
                <td>
                    <div class="tooltip">Erro Percentual Absoluto Médio Simétrico (SMAPE)
                        <span class="tooltiptext">{metric_tooltips["SMAPE"]}</span>
                    </div>
                </td>
                <td>{smape:.2f}%</td>
                <td>
                    <div class="tooltip">{expected_values["SMAPE"][0]:.2f}%
                        <span class="tooltiptext">{expected_values["SMAPE"][1]}</span>
                    </div>
                </td>
            </tr>
            <tr>
                <td>
                    <div class="tooltip">Coeficiente de Determinação (R²)
                        <span class="tooltiptext">{metric_tooltips["R²"]}</span>
                    </div>
                </td>
                <td>{r2:.2f}</td>
                <td>
                    <div class="tooltip">{expected_values["R²"][0]:.2f}
                        <span class="tooltiptext">{expected_values["R²"][1]}</span>
                    </div>
                </td>
            </tr>
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
