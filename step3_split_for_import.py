from pathlib import Path
import re
import pandas as pd


def split_unified_journals_by_asset_month(
    input_path: str = "out/unified_all_journals.csv",
    output_dir: str = "out/ready_for_import",
) -> None:
    """
    Read unified_all_journals.csv and split it into CSVs by asset (ticker) and month.
    Output files are saved in out/ready_for_import.

    Expected columns in input:
        month, account, debit, credit, asset, memo, posting_date, Reference
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    # Basic sanity check
    required_cols = {"asset", "posting_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {missing}")

    # Derive YYYY-MM from posting_date for grouping / filenames
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df["year_month"] = df["posting_date"].dt.to_period("M").astype(str)

    # Drop rows where asset or year_month is missing
    df = df[df["asset"].notna() & df["year_month"].notna()].copy()
    # Group by asset (ticker) and month
    for (asset, year_month), group in df.groupby(["asset", "year_month"], sort=True):
        # Make asset safe for filenames
        safe_asset = re.sub(r"[^A-Za-z0-9_]+", "_", str(asset))

        # Example filename: 2024-01_BTC.csv
        filename = f"{year_month}_{safe_asset}.csv"
        out_path = output_dir / filename

        # Write CSV without index
        group.to_csv(out_path, index=False)

        # Optional: echo what was written (you can remove this in production)
        print(f"Written: {out_path} (rows: {len(group)})")


if __name__ == "__main__":
    split_unified_journals_by_asset_month()