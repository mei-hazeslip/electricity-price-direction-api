# src/predict.py
from pathlib import Path
import joblib
import pandas as pd

from data_transformer import DataTransformer
from features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.pkl"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
OUTPUT_PATH = PROJECT_ROOT / "predictions.csv"


def load_model():
    model_package = joblib.load(MODEL_PATH)
    return model_package["model"], model_package["feature_columns"]


def predict_from_test_csv():
    model, feature_columns = load_model()

    transformer = DataTransformer()
    df = transformer.load_dataset(data_path=TEST_PATH, labels_path=None)

    X = build_features(df)
    X = X[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]

    output = pd.DataFrame(
        {
            "timestamp": df.index,
            "probability_BetLowPrice": probabilities,
        }
    )

    return output



if __name__ == "__main__":
    predictions = predict_from_test_csv()

    predictions.to_csv(OUTPUT_PATH, index=False)

    print("Prediction completed successfully.")
    print(predictions.head())
    print(f"Number of predictions: {len(predictions)}")
    print(f"Saved predictions to {OUTPUT_PATH}")