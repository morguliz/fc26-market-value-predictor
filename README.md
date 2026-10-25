# FC 26 Player Market Value Predictor

A beginner machine learning project that predicts EA Sports FC 26 player valuations using linear regression.

## Overview

This project explores the relationship between a footballer's attributes and their in-game market value.

## Dataset

250 players from EA Sports FC 26.

**Features:**
- Age
- Overall rating
- Potential

**Target:** Player value in euros (`value_eur`).

## Model

- Algorithm: Linear Regression
- Training data: 80% (200 players)
- Testing data: 20% (50 players)
- Random state: 42

## Results

**Mean Absolute Error: €6,736,917.05**

On average, the model's predicted valuations differ from actual in-game values by approximately €6.74 million.

This is an initial baseline model, not a production-ready valuation system.

## Technologies

- Python
- pandas
- scikit-learn

## Future Improvements

- Compare against a simple baseline predictor
- Experiment with additional player attributes
- Compare different regression algorithms
- Evaluate model performance using R² and RMSE
