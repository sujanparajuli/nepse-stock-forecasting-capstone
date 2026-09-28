# NEPSE Stock Forecasting Capstone

Undergraduate capstone studying whether information available at a trading session's close can improve a one-session-ahead forecast for a single NEPSE stock. The first ticker is NABIL. Raw data remains private.

## Current experiment

Inspired by the technical-indicator approach in [Bishal Joshi's NEPSE forecasting project](https://github.com/Bishal-joshi/NEPSE-Advanced-Stock-Forecasting), this experiment computes moving averages, RSI, MACD, recent returns, volatility, price range, and volume ratio. The implementation here is original and targets the **next observed trading session** using Ridge regression. A separate LSTM experiment can follow after the baseline is established.

The CSV should have one ticker only and these columns: `published_date`, `open`, `high`, `low`, `close`, `traded_quantity`. The ShareSansar scraper exports these fields. Keep the CSV locally in `data/raw/`; it is ignored by Git. Dates must parse with pandas, prices must be positive, and duplicate dates cause an error. Data is sorted oldest to newest before features are calculated.

```bash
python -m venv .venv
# Activate .venv for your operating system, then:
pip install -r requirements.txt
python forecast.py data/raw/NABIL.csv
python -m unittest discover -s tests
```

The command prints the test range, mean absolute errors in the CSV's price units, direction accuracy, and a next-session forecast. It writes `data/processed/test_predictions.csv` with input dates, observed next-session dates, actual closes, baseline forecasts, and model forecasts. Keep generated predictions local.

## Evaluation design

- Each row's features use only that session and earlier sessions. The next row's close is the label; the next row's recorded date is the target date for historical evaluation.
- The first 80% of usable labeled rows trains the model; the last 20% tests it. The scaler learns only from training rows. No shuffling or random cross-validation is used.
- The baseline forecasts the next close as today's close. The model is useful only if it improves on that baseline on unseen data. Up/down accuracy counts an unchanged close as "not up."
- After evaluation, the model is fit to all *labeled* history to generate a forecast from the latest available row. This is a forecast for the next trading session, not an asserted calendar date. Exchange holidays and closures require a separate trading calendar.
- The holdout is one fixed period, so results can vary by market regime. Future work: walk-forward evaluation, a direction classifier, multiple tickers, and a sequence model with a comparable split.

The exploratory results from earlier local work (703 test days, baseline MAE 4.18, regression MAE 4.16) have **not** been reproduced by this script and should not be interpreted as its output. Run it on the approved local NABIL CSV before reporting new numbers.

## Project timeline

- Capstone work: 2026–present; repository created September 2026.
- Current implementation: data validation, time-ordered features, next-session target, baseline and Ridge comparison, exported holdout predictions, and a leakage check.

Forecasts are for academic research, not financial advice.
