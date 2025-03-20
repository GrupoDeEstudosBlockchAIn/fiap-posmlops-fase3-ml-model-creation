import pandas as pd
import numpy as np
from app.report.report_generator import predictive_model_performance_metrics_report
from app.src.config import DATA_LAKE_REFINED, MODEL_DIR
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

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
    files = list(DATA_LAKE_REFINED.glob("*.parquet"))
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
