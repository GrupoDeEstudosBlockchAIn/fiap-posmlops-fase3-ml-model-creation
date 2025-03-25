import datetime
from app.dashboard.dashboard_generator import dashboards_cryptocurrency_forecast
from app.src.config import REPORT_DIR

def predictive_model_performance_metrics_report(mae, mse, rmse, smape, r2, df_predictions):
    """Gera um relatório HTML com as métricas de desempenho do modelo preditivo e chama o dashboard."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    report_filename = REPORT_DIR / f"report_{timestamp}.html"
    
    # Renomeia colunas e formata valores
    df_predictions = df_predictions.rename(columns={
        "Symbol": "Símbolo",
        "Name": "Nome",
        "Predicted Price": "Preço Previsto",
        "Price": "Preço Atual"
    })

    # Reordenando as colunas para incluir "Preço Atual"
    df_predictions = df_predictions[["Nome", "Símbolo", "Preço Atual", "Preço Previsto"]]    

    # Formata os valores de preço
    df_predictions["Preço Atual"] = df_predictions["Preço Atual"].apply(lambda x: f"${x:.4f}")
    df_predictions["Preço Previsto"] = df_predictions["Preço Previsto"].apply(lambda x: f"${x:.4f}")
        
    # Chama a função do dashboard passando os dados processados
    dash_filename = dashboards_cryptocurrency_forecast(df_predictions)

    # Valores médios esperados para comparação + Tooltips
    expected_values = {
        "MAE": (0.5, "Quanto menor, melhor"),
        "MSE": (0.3, "Quanto menor, melhor"),
        "RMSE": (0.5, "Quanto menor, melhor"),
        "SMAPE": (10.0, "Ideal abaixo de 10%"),
        "R²": (0.85, "Quanto mais próximo de 1, melhor")
    }

    # Tooltips para as métricas
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
        <title>Relatório de Métricas do Modelo Preditivo</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #2c3e50; color: white; }}
            th.preco-previsto-header {{ background-color: green; color: white; }}
            .highlight-red {{ color: red; font-weight: bold; }}
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
            /* Definição do tamanho das colunas */
            .col-metric {{ width: 34%; }}
            .col-value {{ width: 33%; }}
            .col-expected {{ width: 33%; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Métricas do Modelo Preditivo</h1>
        <p><strong>Data da Previsão:</strong> {now.strftime('%d/%m/%Y')}</p>
        <p><strong>Horário da Previsão:</strong> {now.strftime('%H:%M')}</p>
        
        <h2>Criptomoedas com Maior Potencial de Valorização</h2>
        {df_predictions.to_html(index=False).replace('<th>Preço Previsto</th>', '<th class="preco-previsto-header">Preço Previsto</th>')}
        
        <h2>Métricas de Desempenho</h2>
        <table>
            <tr>
                <th class="col-metric">Métrica</th>
                <th class="col-value">Valor</th>
                <th class="col-expected">Valor Médio Esperado</th>
            </tr>
    """

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "SMAPE": smape,
        "R²": r2
    }

    for metric, value in metrics.items():
        expected = expected_values[metric][0]
        tooltip = metric_tooltips[metric]

        # Define se o valor será vermelho
        if metric == "R²":
            highlight_class = "highlight-red" if value < expected else ""
        else:
            highlight_class = "highlight-red" if value > expected else ""

        report_content += f"""
            <tr>
                <td class="col-metric">
                    <div class="tooltip">{metric}
                        <span class="tooltiptext">{tooltip}</span>
                    </div>
                </td>
                <td class="col-value {highlight_class}">{value:.2f}</td>
                <td class="col-expected">
                    <div class="tooltip">{expected:.2f}
                        <span class="tooltiptext">{expected_values[metric][1]}</span>
                    </div>
                </td>
            </tr>
        """

    report_content += f"""
        </table>
        
        <h2>Dashboard</h2>
        <p><a href="{dash_filename}" target="_blank">Clique aqui para visualizar o Dashboard</a></p>
    </body>
    </html>
    """

    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(report_content)
