# main.py  (FINAL WORKING VERSION for your Kaggle budget.csv)
# Input : raw/budget.csv
# Output: output/union_budget_clean.csv
#         output/sector_year_summary.csv
#         output/ministry_year_summary.csv
#         output/sector_forecast.csv
# Run   : python main.py

import os
import re
import numpy as np
import pandas as pd

RAW_FILE = r"raw\budget.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- helpers ----------------
def to_number(x):
    """Convert values like '1,23,456' or '₹ 123' to float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("₹", "").replace(",", "")
    s = s.replace("cr", "").replace("crore", "")
    s = s.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_measure_col(colname: str):
    """
    Parse columns like:
      'Actuals 2021-2022 Total'
      'Budget Estimates 2022-2023 Revenue'
      'Budget Estimates2023-2024 Capital'
      'Revised Estimates2022-2023 Total'
    Returns (stage, fy, component) or (None, None, None) if not match.
    """
    c = str(colname).replace("\n", " ").strip()
    c = re.sub(r"\s+", " ", c)

    # Find FY like 2021-2022
    mfy = re.search(r"(20\d{2})\s*-\s*(20\d{2})", c)
    if not mfy:
        return (None, None, None)

    fy = f"{mfy.group(1)}-{mfy.group(2)}"

    # Stage
    if c.lower().startswith("actuals"):
        stage = "Actuals"
    elif c.lower().startswith("budget estimates"):
        stage = "Budget Estimates"
    elif c.lower().startswith("revised estimates"):
        stage = "Revised Estimates"
    else:
        stage = "Other"

    # Component
    lc = c.lower()
    if "revenue" in lc:
        component = "Revenue"
    elif "capital" in lc:
        component = "Capital"
    elif "total" in lc:
        component = "Total"
    else:
        component = "Value"

    return (stage, fy, component)


# ---------------- load ----------------
if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(
        f"❌ raw/budget.csv not found.\n"
        f"Put your dataset file here: Internproject\\raw\\budget.csv"
    )

df = pd.read_csv(RAW_FILE)
print("✅ Loaded:", RAW_FILE)
print("Rows:", len(df), "| Columns:", len(df.columns))

# Rename key columns (based on your dataset)
# Your dataset columns look like: Category, Ministry/Department, Scheme, etc.
cat_col = "Category" if "Category" in df.columns else None
min_col = "Ministry/Department" if "Ministry/Department" in df.columns else None
scheme_col = "Scheme" if "Scheme" in df.columns else None

if not cat_col or not min_col:
    print("\n❌ Your CSV does not have expected columns like 'Category' and 'Ministry/Department'.")
    print("Columns found:", list(df.columns))
    raise SystemExit

# ---------------- reshape wide -> long ----------------
id_vars = [c for c in [cat_col, min_col, scheme_col, "Sl.No."] if c and c in df.columns]
value_vars = [c for c in df.columns if c not in id_vars]

long = df.melt(id_vars=id_vars, value_vars=value_vars,
               var_name="Measure", value_name="Allocation")

# Parse measure column into Stage, FY, Component
parsed = long["Measure"].apply(parse_measure_col)
long["Stage"] = parsed.apply(lambda x: x[0])
long["FY"] = parsed.apply(lambda x: x[1])
long["Component"] = parsed.apply(lambda x: x[2])

# Keep only rows where FY was detected
long = long.dropna(subset=["FY"]).copy()

# Numeric allocation
long["Allocation_Cr"] = long["Allocation"].apply(to_number)

# Keep only valid numeric rows
long = long.dropna(subset=["Allocation_Cr"]).copy()

# Add clean standard columns for Tableau
long.rename(columns={
    cat_col: "Sector",
    min_col: "Ministry",
    scheme_col: "Scheme"
}, inplace=True)

# FY start year for sorting
long["FY_StartYear"] = long["FY"].str.slice(0, 4).astype(int)

# Final clean output
clean_cols = ["FY", "FY_StartYear", "Stage", "Component", "Sector", "Ministry", "Scheme", "Allocation_Cr"]
clean = long[clean_cols].copy()

clean_out = os.path.join(OUT_DIR, "union_budget_clean.csv")
clean.to_csv(clean_out, index=False)
print("✅ Saved:", clean_out)
print("Clean rows:", len(clean))

# ---------------- summaries ----------------
# Sector-wise totals per FY (use Total component when available)
sector_year = (clean[clean["Component"] == "Total"]
               .groupby(["FY_StartYear", "FY", "Stage", "Sector"], dropna=False)["Allocation_Cr"]
               .sum().reset_index()
               .sort_values(["FY_StartYear", "Allocation_Cr"], ascending=[True, False]))

sector_year_out = os.path.join(OUT_DIR, "sector_year_summary.csv")
sector_year.to_csv(sector_year_out, index=False)
print("✅ Saved:", sector_year_out)

ministry_year = (clean[clean["Component"] == "Total"]
                 .groupby(["FY_StartYear", "FY", "Stage", "Ministry"], dropna=False)["Allocation_Cr"]
                 .sum().reset_index()
                 .sort_values(["FY_StartYear", "Allocation_Cr"], ascending=[True, False]))

ministry_year_out = os.path.join(OUT_DIR, "ministry_year_summary.csv")
ministry_year.to_csv(ministry_year_out, index=False)
print("✅ Saved:", ministry_year_out)

# ---------------- simple forecast (Sector-wise) ----------------
# We'll forecast based on "Budget Estimates" + "Total" across FYs for each sector
be = sector_year[(sector_year["Stage"] == "Budget Estimates")].copy()
pred_rows = []

for sector, sdf in be.groupby("Sector"):
    sdf = sdf.sort_values("FY_StartYear")
    y = sdf["Allocation_Cr"].to_numpy(dtype=float)
    years = sdf["FY_StartYear"].to_numpy(dtype=int)

    if len(y) == 0:
        continue

    # simple next-year prediction: last + (last - prev) if possible
    if len(y) >= 2:
        pred = y[-1] + (y[-1] - y[-2])
    else:
        pred = y[-1]

    pred_rows.append({
        "Sector": sector,
        "Next_FY_StartYear": int(years.max() + 1),
        "Predicted_Allocation_Cr": round(float(pred), 2),
        "LastKnown_Allocation_Cr": round(float(y[-1]), 2),
        "Years_Used": int(len(y))
    })

forecast = pd.DataFrame(pred_rows).sort_values("Predicted_Allocation_Cr", ascending=False)
forecast_out = os.path.join(OUT_DIR, "sector_forecast.csv")
forecast.to_csv(forecast_out, index=False)
print("✅ Saved:", forecast_out)

print("\n🎉 DONE! Now open Tableau and load:")
print("1) output/union_budget_clean.csv (main)")
print("2) output/sector_year_summary.csv (sector trends)")
print("3) output/ministry_year_summary.csv (ministry trends)")
print("4) output/sector_forecast.csv (prediction)")
