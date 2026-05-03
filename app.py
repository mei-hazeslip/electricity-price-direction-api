# app.py
import joblib
import tempfile
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.data_transformer import DataTransformer
from src.features import build_features
from src.database import init_db, log_prediction, get_recent_logs


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.pkl"

app = FastAPI(
    title="Electricity Price Direction Prediction API",
    description="Predicts the probability that real-time electricity price is lower than day-ahead price.",
    version="0.1.0",
)


def load_model():
    model_package = joblib.load(MODEL_PATH)
    return model_package["model"], model_package["feature_columns"]


model, feature_columns = load_model()
transformer = DataTransformer()
init_db()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Electricity Price Direction Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 0 20px;
                    line-height: 1.6;
                }
                img {
                    max-width: 100%;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    margin-top: 10px;
                    margin-bottom: 20px;
                }
                a.button {
                    display: inline-block;
                    margin-right: 12px;
                    padding: 10px 16px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                }
                code {
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <h1>Electricity Price Direction Prediction API</h1>
            <p>
                This app predicts the probability that the real-time electricity price
                is lower than the day-ahead price.
            </p>


            <h2>Expected Input Format</h2>
            <p>Upload a CSV with the same structure as the original test data.</p>

            <img src="/static/example_input_table.png" alt="Example input table" />

            <h2>Available Endpoints</h2>
            <ul>
                <li><code>POST /predict-csv</code> — upload a CSV and get predictions</li>
                <li><code>GET /logs</code> — view recent SQLite prediction logs</li>
            </ul>
        </body>
    </html>
    """


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with the same format as test.csv.
    Returns prediction probabilities for each timestamp.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    df = transformer.load_dataset(data_path=tmp_path)

    X = build_features(df)
    X = X[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]

    output = pd.DataFrame(
        {
            "timestamp": df.index.astype(str),
            "probability_BetLowPrice": probabilities,
        }
    )

    mean_probability = output["probability_BetLowPrice"].mean()
    min_probability = output["probability_BetLowPrice"].min()
    max_probability = output["probability_BetLowPrice"].max()

    log_prediction(
        filename=file.filename,
        n_predictions=len(output),
        mean_probability=mean_probability,
        min_probability=min_probability,
        max_probability=max_probability,
    )

    return {
        "n_predictions": len(output),
        "probability_summary": {
            "mean": float(mean_probability),
            "min": float(min_probability),
            "max": float(max_probability),
        },
        "sample_predictions": output.head(10).to_dict(orient="records"),
    }

    # return {
    #     "n_predictions": len(output),
    #     "probability_summary": {
    #         "mean": float(output["probability_BetLowPrice"].mean()),
    #         "min": float(output["probability_BetLowPrice"].min()),
    #         "max": float(output["probability_BetLowPrice"].max()),
    #     },
    #     "sample_predictions": output.head(10).to_dict(orient="records"),
    # }


@app.get("/logs")
def logs():
    return {
        "recent_logs": get_recent_logs(limit=10)
    }