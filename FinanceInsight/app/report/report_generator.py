
import datetime
from app.dashboard.dashboard_generator import dashboards_cryptocurrency_forecast
from app.src.config import REPORT_DIR

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

