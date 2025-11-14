#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# ========================
# CONFIG (edit to your needs)
# ========================
INPUT_PATH   = Path("unified_transactions.csv")     # normalized file (CSV or XLSX)
OUTPUT_DIR   = Path("out")                          # where CSV outputs go
WORKBOOK_XLS = Path("crypto_reporting_tool.xlsx")   # Excel report to (re)generate

PARAM_ACCOUNTS = {
    # Adjust to your Odoo Chart of Accounts
    "cash_eur": "101000 Cash EUR",
    "cash_usd_like": "101100 Cash USD/USDT",
    "crypto_inventory_prefix": "1460 Crypto Asset ",
    "fees_expense": "611500 Trading Fees",
    "realized_pnl": "701000 Trading Gains/Losses",
    # ---- Additional accounts for ledger-driven journals ----
    "bank_eur": "101000 Bank EUR",
    "exchange_eur": "101050 Exchange EUR",
    "staking_income": "531200 Staking/Other income",
    "transfers_clearing": "149900 Transfers clearing",
}

# ========================
# Helpers
# ========================
def _force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_unified(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path, low_memory=False, parse_dates=["date_utc"])
    return df

# (FIFO PnL + build_monthly_journal + write_csvs + write_excel_report functions go here —
# keep your working versions unchanged)

# ==================== ADDED: Buys & Ledger Journal Builders ====================

def build_buys_journal(pnl_detailed: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pnl_detailed is None or pnl_detailed.empty:
        return pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])
    df = pnl_detailed[pnl_detailed["side"]=="buy"].copy()
    if df.empty:
        return pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])
    df["month"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.to_period("M").astype(str)

    acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
    acc_fee        = params.get("fees_expense","611500 Trading Fees")
    acc_cash       = params.get("cash_eur", "101000 Cash EUR")

    rows = []
    for _, r in df.iterrows():
        m = r["month"]; asset = r["asset"]
        cost = float(r.get("cost_eur", 0.0) or 0.0)
        fees = float(r.get("fees_eur", 0.0) or 0.0)
        inv_acc = f"{acc_inv_prefix}{asset}"
        if cost>0: rows.append({"month": m, "account": inv_acc, "debit": cost, "credit": 0.0, "asset": asset, "memo": f"Buy {asset} - inventory at cost"})
        if fees>0: rows.append({"month": m, "account": acc_fee, "debit": fees, "credit": 0.0, "asset": asset, "memo": f"Buy {asset} - fees"})
        outflow = cost
        if outflow>0: rows.append({"month": m, "account": acc_cash, "debit": 0.0, "credit": outflow, "asset": asset, "memo": f"Buy {asset} - cash outflow"})

    jl = pd.DataFrame(rows)
    if jl.empty: return jl
    for c in ["debit","credit"]:
        jl[c] = pd.to_numeric(jl[c], errors="coerce").fillna(0.0)
    jl = jl.groupby(["month","account","asset","memo"], as_index=False).agg(debit=("debit","sum"), credit=("credit","sum"))
    jl["debit"] = jl["debit"].round(2); jl["credit"] = jl["credit"].round(2)
    return jl

def build_ledger_journals(unified: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = unified.copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df["month"] = df["date_utc"].dt.to_period("M").astype(str)
    led = df[(df["source"]=="ledger") & (df["event_type"].str.lower()!="trade")].copy()
    led["asset"] = led["base_ccy"].astype(str).str.upper()
    led["etype"] = led["event_type"].astype(str).str.lower()

    acc_bank_eur   = params.get("bank_eur", "101000 Bank EUR")
    acc_exch_eur   = params.get("exchange_eur", params.get("cash_eur","101000 Cash EUR"))
    acc_exch_usdl  = params.get("exchange_usd_like", params.get("cash_usd_like","101100 Cash USD/USDT"))
    acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
    acc_fee        = params.get("fees_expense","611500 Trading Fees")
    acc_income     = params.get("staking_income","531200 Staking/Other income")
    acc_clear      = params.get("transfers_clearing","149900 Transfers clearing")

    rows = []
    def add(m, acc, dr, cr, asset, memo):
        rows.append({"month": m, "account": acc, "debit": float(dr or 0.0), "credit": float(cr or 0.0), "asset": asset, "memo": memo})

    for _, r in led.iterrows():
        m = r["month"]; et = r["etype"]; asset = r["asset"]
        eur_amt = float(r["eur_amount"]) if pd.notna(r["eur_amount"]) else 0.0
        fee = float(r["eur_fee"]) if pd.notna(r["eur_fee"]) else 0.0

        # handle EUR deposits/withdrawals, USD-like, crypto transfers, staking income, and fees
        # (implementation omitted here for brevity — use the version I generated earlier)

    jl = pd.DataFrame(rows)
    if jl.empty: return jl
    for c in ["debit","credit"]:
        jl[c] = pd.to_numeric(jl[c], errors="coerce").fillna(0.0)
    jl = jl.groupby(["month","account","asset","memo"], as_index=False).agg(debit=("debit","sum"), credit=("credit","sum"))
    jl["debit"] = jl["debit"].round(2); jl["credit"] = jl["credit"].round(2)
    return jl

# ==================== END ADDED ====================

def main():
    unified = load_unified(INPUT_PATH)
    # your existing fifo_pnl, build_monthly_journal, write_csvs, write_excel_report go here
    # ...
    # After your existing Excel/CSV writes, add:

    # === Extra journals: buys & ledger ===
    try:
        jl_buys = build_buys_journal(pnl_detailed, PARAM_ACCOUNTS)
    except Exception as _e:
        print("WARN: buys journal failed:", _e); jl_buys = pd.DataFrame()

    try:
        jl_ledger = build_ledger_journals(unified, PARAM_ACCOUNTS)
    except Exception as _e:
        print("WARN: ledger journal failed:", _e); jl_ledger = pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not jl_buys.empty:
        (OUTPUT_DIR/"journal_buys_monthly.csv").write_text(jl_buys.to_csv(index=False))
    if not jl_ledger.empty:
        (OUTPUT_DIR/"journal_ledger_monthly.csv").write_text(jl_ledger.to_csv(index=False))

    # Combine with existing trade journal if desired
    # ...

    print("Done.")

if __name__ == "__main__":
    main()
