# Bank Marketing: Customer Response Prediction

## What this project does
This OEP uses classification to estimate whether a customer contacted during a bank marketing campaign subscribed to a term deposit.

## Algorithms
- Decision Tree
- Logistic Regression

Both algorithms are trained on the same training split and evaluated on a held-out test split. The notebook includes preprocessing, model evaluation and a comparison of the results.

## Files
- `main.ipynb` — data inspection, preprocessing, model training, evaluation and comparison.
- `app.py` — Streamlit interface for trying a customer record and comparing model scores.
- `bank.csv` — dataset used by the project.
- `model_comparison.csv` — saved model comparison from the notebook.

## Run the project
Install the libraries if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

Place the files in the same folder, then run:

```bash
streamlit run app.py
```

Open `main.ipynb` in Jupyter Notebook or VS Code to run the analysis.

## Important modelling choice
The `duration` column is excluded from the predictors because call duration is only known after the call. Including it when deciding whom to contact beforehand could make the model less useful for planning a campaign.

## Limitations
The prediction is an estimate from historical data. It should not be treated as a guarantee or used as the only basis for a real marketing decision. Model scores can also change with a different train/test split or preprocessing choices.
