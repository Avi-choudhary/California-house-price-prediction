California House Price Prediction API

A FastAPI practice project built to learn REST API design and model-serving patterns, as part of my transition from Python backend development into machine learning.

What this is

This project wraps a scikit-learn regression model in a FastAPI service. The focus isn't the model itself (it's trained on the standard California housing dataset) — it's the API layer: request validation, error handling, single vs. batch prediction endpoints, and structuring a small backend service end to end.

Stack

- FastAPI — REST API framework
- scikit-learn — RandomForestRegressor for the underlying model
- Pandas — data handling
- Pydantic — request validation
- joblib — model persistence

Project structure

```
explore.py   # initial data exploration (shape, summary stats)
train.py     # trains the model, evaluates it, saves model + feature list
main.py      # FastAPI app serving the trained model
```

Model performance

Trained with an 80/20 train-test split, `RandomForestRegressor(n_estimators=100, random_state=42)`:

- MAE: ~$39,000
- Evaluated with `mean_absolute_error` and `r2_score` on the held-out test set

API endpoints

`GET /`
Health/info endpoint — confirms the API is running.

`GET /health`
Returns model status, feature list, and average error.

`POST /predict`
Predicts a price for a single house given its features (median income, house age, average rooms/bedrooms, population, average occupancy, latitude, longitude). Inputs are validated with Pydantic (e.g. latitude/longitude bounded to California's range).

`POST /predict-file`
Accepts a CSV upload with the required feature columns and returns predictions for every row as a downloadable CSV.

Running locally

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
python train.py        # trains and saves the model
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive API docs.

Why this project

I'm learning FastAPI as a backend skill while working toward machine learning. This project is a stepping stone — a dedicated, more original project (custom dataset, deployed, tested) is planned as a follow-up.
