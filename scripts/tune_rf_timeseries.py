import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
GRID_FILE = DATA_DIR / "rf_tuning_grid.csv"
FINAL_FILE = DATA_DIR / "rf_tuning_final.csv"

clean = pd.read_csv(DATA_DIR / "nigeria_covid_clean.csv", parse_dates=["date"])
features = pd.read_csv(DATA_DIR / "nigeria_covid_features.csv", parse_dates=["date"])
clean = clean.sort_values("date").reset_index(drop=True)
features = features.sort_values("date").reset_index(drop=True)

clean["roll7_mean"] = clean["new_cases"].rolling(7).mean()
clean["lag_1_cases"] = clean["new_cases"].shift(1)
clean["target_base"] = clean["new_cases"].shift(-1)
base = clean[["date", "lag_1_cases", "target_base", "roll7_mean"]]

ENG = ["new_cases_lag_1", "new_cases_lag_7", "new_deaths_lag_1", "new_deaths_lag_7",
       "new_cases_rolling_7", "new_deaths_rolling_7", "cases_change_1d",
       "deaths_change_1d", "day_of_week", "month"]
TARGET = "target_next_day_cases"

df = base.merge(features[["date", TARGET] + ENG], on="date", how="inner")
df = df.dropna().sort_values("date").reset_index(drop=True)
if not np.allclose(df["target_base"], df[TARGET]):
    raise ValueError("Target values do not match between baseline and engineered data.")

folds = list(TimeSeriesSplit(n_splits=5, gap=7).split(df))
SELECT_FOLDS = (1, 2, 3)   # used to choose parameters
HOLDOUT_FOLDS = (4, 5)     # never used for choosing parameters

def score(y, p):
    return mean_absolute_error(y, p), float(np.sqrt(mean_squared_error(y, p)))

ref = {}
for i, (tr, te) in enumerate(folds, start=1):
    test = df.iloc[te]
    ref[i] = {"predict_zero": score(test[TARGET].values, np.zeros(len(test))),
              "mean_7day": score(test[TARGET].values, test["roll7_mean"].values)}

def run(cols, params):
    out = {}
    for i, (tr, te) in enumerate(folds, start=1):
        train, test = df.iloc[tr], df.iloc[te]
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, **params)
        m.fit(train[cols], train[TARGET])
        out[i] = score(test[TARGET].values, m.predict(test[cols]))
    return out

MODELS = {"baseline": ["lag_1_cases"], "engineered": ENG}
LEAF, DEPTH, MAXF = [1, 5, 20], [None, 8], [1.0, 0.5]

records = []
for name, cols in MODELS.items():
    mf_options = MAXF if len(cols) > 1 else [1.0]
    for leaf, depth, mf in itertools.product(LEAF, DEPTH, mf_options):
        params = dict(min_samples_leaf=leaf, max_depth=depth, max_features=mf)
        res = run(cols, params)
        sel = np.mean([0.5 * (res[i][0] / ref[i]["mean_7day"][0] +
                              res[i][1] / ref[i]["mean_7day"][1]) for i in SELECT_FOLDS])
        rec = {"model": name, "min_samples_leaf": leaf,
               "max_depth": "None" if depth is None else depth,
               "max_features": mf, "selection_score": sel}
        for i in res:
            rec[f"MAE_f{i}"], rec[f"RMSE_f{i}"] = res[i]
        records.append(rec)
        print(f"{name:10s} leaf={leaf:<3} depth={str(depth):<5} maxf={mf}  score={sel:.3f}")

grid = pd.DataFrame(records)
grid.to_csv(GRID_FILE, index=False)

def pick(name, tuned):
    g = grid[grid["model"] == name]
    if tuned:
        return g.loc[g["selection_score"].idxmin()]
    return g[(g["min_samples_leaf"] == 1) & (g["max_depth"] == "None") & (g["max_features"] == 1.0)].iloc[0]

chosen = {"rf_baseline_default": pick("baseline", False), "rf_baseline_tuned": pick("baseline", True),
          "rf_engineered_default": pick("engineered", False), "rf_engineered_tuned": pick("engineered", True)}

print("\nSELECTED PARAMETERS (chosen on folds 1-3 only)")
for k in ("rf_baseline_tuned", "rf_engineered_tuned"):
    r = chosen[k]
    print(f"{k}: min_samples_leaf={r['min_samples_leaf']}, max_depth={r['max_depth']}, max_features={r['max_features']}")

final = {}
final["predict_zero"] = {f"MAE_f{i}": ref[i]["predict_zero"][0] for i in ref} | {f"RMSE_f{i}": ref[i]["predict_zero"][1] for i in ref}
final["mean_7day"] = {f"MAE_f{i}": ref[i]["mean_7day"][0] for i in ref} | {f"RMSE_f{i}": ref[i]["mean_7day"][1] for i in ref}
for k, r in chosen.items():
    final[k] = {c: r[c] for c in grid.columns if c.startswith(("MAE_", "RMSE_"))}

order = ["predict_zero", "mean_7day", "rf_baseline_default", "rf_baseline_tuned",
         "rf_engineered_default", "rf_engineered_tuned"]
out_rows = []
for metric in ("MAE", "RMSE"):
    t = pd.DataFrame({m: [final[m][f"{metric}_f{i}"] for i in range(1, 6)] for m in order},
                     index=[1, 2, 3, 4, 5]).T
    t["avg_f1-3"] = t[[1, 2, 3]].mean(axis=1)
    t["avg_f4-5 (holdout)"] = t[[4, 5]].mean(axis=1)
    print(f"\n{metric} by fold")
    print(t.round(2).to_string())
    t2 = t.copy(); t2.insert(0, "metric", metric)
    out_rows.append(t2.reset_index().rename(columns={"index": "model"}))

    b, e = t.loc["rf_baseline_tuned"], t.loc["rf_engineered_tuned"]
    pct = ((b - e) / b * 100).round(1)
    print(f"{metric}: engineered_tuned vs baseline_tuned, % improvement (positive = better)")
    print(pct.to_string())

pd.concat(out_rows).to_csv(FINAL_FILE, index=False)
print("\nSaved:", GRID_FILE, "and", FINAL_FILE)
