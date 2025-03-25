import pandas as pd
import numpy as np
from app.report.report_generator import predictive_model_performance_metrics_report
from app.src.config import MODEL_DIR
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def filtrar_criptomoedas_em_alta(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Filtra as criptomoedas que possuem previsão de alta e retorna as top N com maior potencial.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo as colunas 'Price' (PREÇO_ATUAL) e 'Predicted Price' (PREÇO_PREVISTO).
    top_n (int): Número de criptomoedas a exibir.

    Retorna:
    pd.DataFrame: DataFrame com as criptomoedas que têm previsão de alta, ordenadas pelo maior potencial de valorização.
    """
    df["DIFERENÇA"] = df["Predicted Price"] - df["Price"]
    df_alta = df[df["DIFERENÇA"] > 0]  # Filtra apenas criptos com previsão de alta
    df_alta = df_alta.sort_values(by="DIFERENÇA", ascending=False)  # Ordena pelas maiores diferenças
    return df_alta.head(top_n)  # Retorna as top N criptomoedas com maior potencial

# Fazer previsões
def best_model_predict(best_model, scaler: StandardScaler, X_test: pd.DataFrame, y_test, df: pd.DataFrame, features):    
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    smape = (2 * abs(y_test - y_pred) / (abs(y_test) + abs(y_pred))).mean() * 100
    r2 = r2_score(y_test, y_pred)
        
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = best_model.predict(pd.DataFrame(scaler.transform(df[features].iloc[-10:]), columns=features))    

    # Aplicando o filtro para mostrar apenas criptomoedas com previsão de alta
    df_top_alta = filtrar_criptomoedas_em_alta(df_predictions)

    # Relatório de métricas
    predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_top_alta[["Symbol", "Predicted Price", "Name", "Price"]])
    
    # Salvando modelo e scaler
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
