---
title: Electricity Price Direction API
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---
# Electricity Price Spread Direction Prediction API

> **Check out the [interactive project page](https://mei-hazeslip.github.io/electricity-trading.html) for a simple breakdown and project summary.**

This project predicts the hourly direction of electricity price spreads between the day-ahead market and the real-time market. The target is whether the real-time price is lower than the day-ahead price, which can help electricity buyers evaluate potential bidding and procurement strategies.

The original work began as a machine learning notebook and was refactored into a deployable FastAPI application with reusable preprocessing, feature engineering, model training, and prediction scripts.

## Project Objective

Companies with high electricity consumption often need to decide how much electricity to commit to in the day-ahead market versus how much exposure to leave for the real-time market. If the real-time price is lower than the day-ahead price, there may be an opportunity to reduce costs.

The goal of this project is to predict:

```text
1 = Real-time price < Day-ahead price
0 = Real-time price > Day-ahead price
```

## Deployment Overview

This repository contains a lightweight model-serving API built with FastAPI. Users can upload a CSV file with the same schema as the test data, and the API returns predicted probabilities for whether the real-time electricity price will be lower than the day-ahead price.

Current workflow:

```text
CSV upload
→ data transformation
→ feature engineering
→ LightGBM prediction
→ probability summary and sample predictions
```

## Project Structure

```text
electricity-price-api/
├── app.py                  # FastAPI application
├── requirements.txt        # Python dependencies
├── models/
│   └── lgbm_model.pkl      # Saved LightGBM model artifact
├── src/
│   ├── data_transformer.py # Data loading and column renaming
│   ├── features.py         # Feature engineering pipeline
│   ├── train_model.py      # Model training script
│   └── predict.py          # Local batch prediction script
└── notebooks/
    └── power_quant.ipynb   # Original exploratory modeling notebook
```

## Model

The deployed model is a LightGBM classifier trained on engineered electricity-market features, including:

- day-ahead forecast variables
- real-time lagged market variables
- price-memory features
- 24-hour and 48-hour ramp features
- renewable/load mismatch features
- hour-of-day interaction features

The model outputs a probability that the real-time price is lower than the day-ahead price.

## Running Locally

Install dependencies:


```bash
pip install -r requirements.txt
```

Run the FastAPI app:
```bash
python -m uvicorn app:app --reload
```
Open the API documentation page:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /predict-csv` endpoint to upload a CSV file and return prediction results.


## Data Availability

The original electricity-market training data is not included in this repository due to data privacy and usage restrictions. This repository contains the deployment code, feature engineering pipeline, trained model artifact, and a small synthetic example input for demonstration purposes.

## Future Improvements

Planned improvements include:

deploying the API to Hugging Face Spaces
adding SQLite-based prediction logging
adding a simple web interface for CSV upload
improving model monitoring and input validation