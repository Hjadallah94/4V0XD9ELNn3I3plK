from pathlib import Path
import argparse
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

def load_best_model():
    candidates = sorted(MODELS_DIR.glob("*_best_model.joblib"))
    if not candidates:
        raise FileNotFoundError("No saved model in models/. Run training first.")
    return joblib.load(candidates[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with features X1..X6 (Y optional)")
    ap.add_argument("--output", default=str(REPORTS_DIR / "predictions.csv"))
    args = ap.parse_args()

    model = load_best_model()
    df = pd.read_csv(args.input)
    X_cols = [c for c in df.columns if c != "Y"]
    preds = model.predict(df[X_cols])

    out = df.copy()
    out["prediction"] = preds
    try:
        out["proba_happy"] = model.predict_proba(df[X_cols])[:, 1]
    except Exception:
        pass

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
