# Women’s March Madness Prediction Pipeline (V2)

## Overview
This project builds a women’s NCAA tournament prediction pipeline optimized for log loss. The current best performer is a **calibrated logistic regression**, which outperforms the ensemble given the high correlation between model outputs. The pipeline keeps long-history features (1998+) and adds richer efficiency and external ratings from 2010+.

## Data Sources
- Kaggle Women’s NCAA files in `data/` and `data/raw/`
- ESPN poll votes: `data/espn_ratings/wncaa_espn.csv`
- FiveThirtyEight ratings: `data/fivethirtyeight_ratings/538ratingsWomen.csv`

## Core Feature Groups
1. **Elo (1998–present)**
   - Margin-of-victory Elo with offseason regression
2. **Efficiency (2010–present)**
   - Off/Def rating, Net rating, eFG%, TS%, rebounding, turnover, assist ratio
3. **Recent form**
   - Last-5 and last-10 net rating, trend (Last5 – Last10)
4. **Seeds & Upset Modeling**
   - Seed difference, classic upset bands, historical upset rates by seed matchup
   - ESPN/538 seed residuals and residual interactions
5. **Strength of Schedule & Conference Context**
   - Opponent Elo and NetRtg averages
   - Conference Elo
6. **Head-to-Head**
   - Win % and margin vs same opponent (regular season)

## Modeling & Calibration
- **Primary model:** Calibrated Logistic Regression (isotonic calibration)
- **Secondary model:** Random Forest (calibrated)
- **Stacking** is available but currently underperforms LR due to high correlation

## Validation Strategy
- Train: 2010–2021
- Validation: 2022–2023
- Holdout: 2024–2025

## Outputs
- Submission (Kaggle format): `submissions/submission.csv`
- Pairs only: `submissions/WNCAATourneyPredictions.csv`
- Pairs with probabilities: `submissions/WNCAATourneyPredictions_with_preds.csv`
- Championship total estimate: `submissions/championship_total.csv`
- Model metrics: `data/processed/model_metrics_all_models.csv`

## How to Run
1. Build dataset:
   - `python -m src.pipeline.build_dataset`
2. Train models:
   - `python -m src.pipeline.train_models`
3. Generate predictions:
   - `python -m src.pipeline.generate_predictions`
4. Championship total:
   - `python -m src.pipeline.predict_championship_total`
