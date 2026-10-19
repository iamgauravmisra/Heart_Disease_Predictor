# Heart Disease Prediction App

Simple Streamlit app that predicts heart disease using a pre-trained logistic regression model.

## Files

- `app.py` - Streamlit application.
- `LR_heart.pkl` - Trained logistic regression model (joblib).
- `scaler.pkl` - StandardScaler used to scale inputs.
- `columns.pkl` - Ordered list of feature column names expected by the model.
- `heart.csv` - (optional) dataset used for training / examples.

## Setup

1. (Recommended) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install streamlit scikit-learn pandas numpy scipy matplotlib seaborn joblib
```

If you use `conda`:

```bash
conda create -n heart-env python=3.10
conda activate heart-env
conda install -c conda-forge scikit-learn pandas numpy scipy matplotlib seaborn joblib streamlit
```

## Run

From the project directory run either:

```bash
streamlit run app.py
```

or

```bash
python3 -m streamlit run app.py
```

If you see the error `/usr/local/bin/python3: can't open file '/path/to/streamlit'`: use the `-m` form above.

## Troubleshooting

- If predictions look wrong (e.g., always `No Heart Disease`), ensure `app.py` maps UI inputs to the same feature names in `columns.pkl`. You can inspect the expected columns with:

```bash
python3 - <<'PY'
import joblib
print(joblib.load('columns.pkl'))
PY
```

- If unpickling warnings appear about scikit-learn versions, either install the scikit-learn version used to save the model (example: `pip install scikit-learn==1.6.1`) or retrain the model with your current scikit-learn.

- To see model probabilities in the app, `app.py` prints the predicted probability when available. Use extreme/known examples to validate behavior.

## Next steps / Ideas

- Add preset example buttons (low/high risk) in the UI for quick testing.
- Provide a `requirements.txt` or `pyproject.toml` to lock dependency versions.
- Retrain or calibrate the model if you need different sensitivity/precision trade-offs.

---

Created to help run and debug the Streamlit heart-disease demo.
