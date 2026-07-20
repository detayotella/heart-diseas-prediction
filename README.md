# Heart Disease Prediction Dashboard

A clinical-style Streamlit dashboard for estimating heart disease risk using a trained **Random Forest** model.

This project includes:
- a reusable Python package in `src/heart_disease`
- a pretrained model in `models/heart_disease_rf.pkl`
- a Streamlit UI in `app.py`
- a training notebook in `notebooks/01-random_forest.ipynb`

> ⚠️ **Disclaimer**: This project is for educational/research use only. It is **not** a medical diagnosis tool.

---

## Features

- Interactive patient input form (demographics, ECG/exercise findings, lab/vitals)
- Predicted class (`Heart Disease` or `No Heart Disease`)
- Risk-level labeling (`Low`, `Medium`, `High`)
- Probability breakdown and visual risk bar
- Clinical-style dashboard layout and summary table

---

## Project Structure

```text
heart-disease-prediction/
├── app.py
├── pyproject.toml
├── uv.lock
├── data/
│   ├── raw/heart_data_set.csv
│   └── preprocessed/
├── models/
│   └── heart_disease_rf.pkl
├── notebooks/
│   └── 01-random_forest.ipynb
└── src/
    └── heart_disease/
        ├── __init__.py
        ├── predict.py
        ├── preprocessing.py
        ├── train.py
        └── utils.py
```

---

## Requirements

- **Python 3.14+** (as specified in `pyproject.toml`)
- Linux/macOS/Windows

Main dependencies:
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`

---

## Setup

### Option A: Using `uv` (recommended)

```bash
uv sync
```

Then run commands with the managed environment, e.g.:

```bash
uv run streamlit run app.py
```

### Option B: Using virtualenv + pip

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Data and Model

- Raw dataset expected at: `data/raw/heart_data_set.csv`
- Default model path: `models/heart_disease_rf.pkl`

The app loads the model through:
- `heart_disease.predict.load_model()`

---

## Training / Experimentation

For model development and retraining, use the notebook:

- `notebooks/01-random_forest.ipynb`

It contains the end-to-end experimentation workflow used for this project.

---

## Using the Package in Python

Example inference usage:

```python
from heart_disease.predict import load_model, predict_from_inputs, get_risk_level

model = load_model()

result = predict_from_inputs(
    model=model,
    age=50,
    sex=1,
    cp=2,
    trestbps=120,
    chol=200,
    fbs=0,
    restecg=1,
    thalach=150,
    exang=0,
    oldpeak=1.0,
    slope=2,
    ca=0,
    thal=3,
)

print(result)
print(get_risk_level(result["probability"]))
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'heart_disease'`
- Make sure dependencies are installed and you are in the project environment.
- Prefer `pip install -e .` (or `uv sync`) so `src/heart_disease` is importable.

### Streamlit/Arrow dataframe warnings
- Ensure mixed-type display columns are cast to string before rendering.
- Restart Streamlit after code changes to avoid stale process state.

### Missing model file
If `models/heart_disease_rf.pkl` is missing, restore/retrain and save the model to that path.

---

## Notes

- This repository currently focuses on local development and interactive use.
- If you want production deployment, consider adding:
  - model/version metadata
  - input validation and schema checks
  - CI tests
  - containerization (`Dockerfile`)

---

## License

No license file is currently present in this repository. Add a `LICENSE` file if you plan to distribute this project.
