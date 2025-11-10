import json
from pathlib import Path
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
DATASET_DIR = BASE_DIR / "data" / "datasets"

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "ITFF Dashboard"

symbols = sorted({p.name.split("_")[0].upper() for p in REPORT_DIR.glob("*_evaluation.json")})
labels = ["direction", "level_up"]
model_types = ["lstm", "transformer"]


def load_metrics(symbol: str, label: str, model_type: str) -> Dict:
    metrics_path = REPORT_DIR / f"{symbol.lower()}_{label}_{model_type}_evaluation.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def load_thresholds(symbol: str, label: str, model_type: str):
    import pandas as pd

    csv_path = REPORT_DIR / f"{symbol.lower()}_{label}_{model_type}_thresholds.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_curve(symbol: str, label: str, model_type: str):
    import numpy as np

    dataset_prefix = f"{symbol.lower()}_test"
    X = np.load(DATASET_DIR / f"{dataset_prefix}_X.npy")
    y = np.load(DATASET_DIR / f"{dataset_prefix}_{label}.npy")
    metrics = load_metrics(symbol, label, model_type)
    if not metrics:
        return None, None
    probs = metrics.get("classification_report", {}).get("support")
    return y, probs

app.layout = dbc.Container(
    [
        html.H1("Intelligent Trading Forecast Dashboard"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(symbols, symbols[0] if symbols else None, id="symbol"),
                        dcc.Dropdown(labels, labels[0], id="label"),
                        dcc.Dropdown(model_types, model_types[0], id="model"),
                        html.Br(),
                        html.Div(id="metric-cards"),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="curve-figure"),
                        dcc.Graph(id="threshold-figure"),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("metric-cards", "children"),
    Output("curve-figure", "figure"),
    Output("threshold-figure", "figure"),
    Input("symbol", "value"),
    Input("label", "value"),
    Input("model", "value"),
)
def update_dashboard(symbol, label, model_type):
    metrics = load_metrics(symbol, label, model_type)
    if not metrics:
        return html.Div("No metrics available. Run evaluation script."), go.Figure(), go.Figure()

    cards = dbc.Row(
        [
            dbc.Col(dbc.Card([dbc.CardHeader("Accuracy"), dbc.CardBody(f"{metrics['accuracy']:.3f}")])),
            dbc.Col(dbc.Card([dbc.CardHeader("Precision"), dbc.CardBody(f"{metrics['precision']:.3f}")])),
            dbc.Col(dbc.Card([dbc.CardHeader("Recall"), dbc.CardBody(f"{metrics['recall']:.3f}")])),
            dbc.Col(dbc.Card([dbc.CardHeader("ROC AUC"), dbc.CardBody(f"{metrics['roc_auc']:.3f}")])),
            dbc.Col(dbc.Card([dbc.CardHeader("Brier"), dbc.CardBody(f"{metrics['brier_score']:.3f}")])),
        ],
        className="g-2",
    )

    thresholds = load_thresholds(symbol, label, model_type)
    threshold_fig = go.Figure()
    if thresholds is not None:
        threshold_fig = px.line(thresholds, x="threshold", y=["precision", "recall", "f1"], title="Threshold Sweep")

    curve_fig = go.Figure()
    curve_path = REPORT_DIR / f"{symbol.lower()}_{label}_{model_type}_curves.png"
    if curve_path.exists():
        curve_fig.add_layout_image(
            dict(source=str(curve_path), xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1, sizing="stretch")
        )
        curve_fig.update_xaxes(showgrid=False, visible=False)
        curve_fig.update_yaxes(showgrid=False, visible=False)
        curve_fig.update_layout(title="ROC / PR / Calibration")

    return cards, curve_fig, threshold_fig


if __name__ == "__main__":
    app.run(debug=True)
