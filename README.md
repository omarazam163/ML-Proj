# Loan Approval Prediction — Decision Tree + SVM + Streamlit

End-to-end ML project: predict loan approval using a Decision Tree (primary, for interpretability) and an SVM (for comparison), with a Streamlit interface for live predictions.

## Setup

```powershell
pip install -r requirements.txt
```

### Get the dataset (Kaggle API)
1. Go to https://www.kaggle.com/settings → **Create New API Token**. This downloads `kaggle.json`.
2. Move it to `C:\Users\<you>\.kaggle\kaggle.json`.
3. Visit the dataset page in your browser and accept the rules.
4. Download:

```powershell
kaggle datasets download -d altruistdelhi/loan-data -p data/raw --unzip
# fallback if the slug above 404s:
# kaggle datasets download -d ninzaami/loan-predication -p data/raw --unzip
```

If both fail, manually download the train CSV and place it at `data/raw/train.csv`.

## Train

Open and run all cells:

```powershell
jupyter notebook notebooks/loan_approval.ipynb
```

This produces `models/dt_pipeline.joblib` and `models/svm_pipeline.joblib`.

## Run the app

```powershell
streamlit run app/streamlit_app.py
```
