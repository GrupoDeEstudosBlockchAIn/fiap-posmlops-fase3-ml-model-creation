import pandas as pd
import numpy as np
from app.report.report_generator import predictive_model_performance_metrics_report
from app.src.config import MODEL_DIR
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fazer previsões
def best_model_predict(best_model, scaler: StandardScaler, X_test: pd.DataFrame, y_test, df: pd.DataFrame, features):    
    
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    smape = (2 * abs(y_test - y_pred) / (abs(y_test) + abs(y_pred))).mean() * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.2f}%, R²: {r2:.4f}")
    
    df_predictions = df.iloc[-10:].copy()
    df_predictions["Predicted Price"] = best_model.predict(pd.DataFrame(scaler.transform(df[features].iloc[-10:]), columns=features))
    
    # Verificando a importância das features
    feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print("Importância das Features:")
    print(feature_importances)
    
    # Relatório de métricas
    predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_predictions[["Symbol", "Predicted Price", "Name"]])
    
    # Salvando modelo e scaler
    model_path = MODEL_DIR / "crypto_price_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")
    
    print("Top 10 previsões de preços:")
    print(df_predictions[["Symbol", "Predicted Price"]])
