# FreshTrack-ShelfLife — Food Shelf-Life Estimation Prototype

**Sector:** Food systems  
**Task:** Regression: synthetic remaining shelf life in days

Estimates remaining shelf life from product class, temperature exposure, humidity, initial moisture, packaging, and handling time. Batches are grouped in the holdout.

## Data and split

The deterministic offline generator in `freshtrack/data.py` creates synthetic examples and uses no external dataset. Group IDs are not model features; the holdout keeps all rows from each route or batch together.

## Methods and metrics

- **Baseline:** training-mean predictor.
- **Stronger model:** histogram gradient-boosting regression.
- **Metrics:** MAE, RMSE, R², 90th-percentile absolute error, and coverage within 2 days.
- **Error analysis:** tail absolute error and tolerance coverage are emitted for both models.

## Limitations

Shelf life depends on validated assays, microbes, storage history, packaging, and safety standards that this simulator omits. Never use for food safety, labeling, or disposal decisions.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m freshtrack.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## Code license status

No code license has been selected for this new project; default copyright rules apply. Choose a license before public distribution. No external data is included.
