# src/train-model.py
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier
from features import build_features
from data_transformer import DataTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"


def train_model():
    transformer = DataTransformer()

    df = transformer.load_dataset(
        data_path=PROJECT_ROOT / "data" / "train.csv",
        labels_path=PROJECT_ROOT / "data" / "labels.csv",
    )

    X = build_features(df)
    y = df["target"]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )

    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": list(X.columns),
    }

    joblib.dump(model_package, MODEL_PATH)

    print("Model trained successfully.")
    print(f"Training rows: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()