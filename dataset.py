"""
dataset.py
Utilities for loading the raw dataset into a pandas DataFrame.
"""

from pathlib import Path
import pandas as pd

# dataset.py is in ...\Happy_Customers\dataset.py
# Project root is ONE level up from Happy_Customers → parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "ACME-HappinessSurvey2020.csv"

def load_raw():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df