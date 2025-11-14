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
    # Required columns
    exp = ["date_utc","source","event_type","base_ccy","quote_ccy","side",
           "qty_base","qty_quote","eur_amount","eur_fee","txid","exchange"]
    missing = [c for c in exp if c not in df.columns]
    if missing:
        raise RuntimeError(f"Unified file missing columns: {missing}")
    # Normalize
    df["side"] = df["side"].astype(str).str.lower()
    for c in ["qty_base","qty_quote","eur_amount","eur_fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fifo_pnl(trades: pd.DataFrame):
    t = trades.copy()
    t = t[(t["source"]=="trades") & (t["event_type"]=="trade")]
    t = t[t["eur_amount"].notna()]
    t = t.sort_values(["date_utc","txid"]).reset_index(drop=True)

    lots = {}
    rows = []

    def add_lot(asset, units, total_cost):
        lots.setdefault(asset, [])
        lots[asset].append({
            "units": float(units),
            "total_cost": float(total_cost),
            "unit_cost": float(total_cost)/float(units) if units else 0.0
        })

    def relieve_lot(asset, units_to_relieve):
        cost = 0.0
        remaining = float(units_to_relieve)
        L = lots.get(asset, [])
        i = 0
        while remaining > 1e-18 and i < len(L):
            lot = L[i]
            take = min(lot["units"], remaining)
            part_cost = take * lot["unit_cost"]
            cost += part_cost
            lot["units"] -= take
            lot["total_cost"] -= part_cost
            remaining -= take
            if lot["units"] <= 1e-18:
                L.pop(i)
            else:
                i += 1
        lots[asset] = L
        return cost, remaining

    for _, r in t.iterrows():
        asset = str(r["base_ccy"]).upper()
        qty_base = float(r["qty_base"] or 0.0)
        fee_eur  = float(r["eur_fee"] or 0.0)
        eur_val  = float(r["eur_amount"] or 0.0)  # quote leg value (sign follows qty_quote)

        if r["side"] == "buy" and qty_base > 0:
            total_cost = -eur_val + fee_eur  # eur_val negative for buys
            add_lot(asset, qty_base, total_cost)
            rows.append({
                "date_utc": r["date_utc"], "exchange": r["exchange"], "txid": r["txid"],
                "asset": asset, "side": "buy", "qty": qty_base,
                "proceeds_eur": 0.0, "cost_eur": total_cost, "fees_eur": fee_eur,
                "realized_pnl_eur": 0.0
            })
        elif r["side"] == "sell" and qty_base < 0:
            units_to_sell = -qty_base
            proceeds_net  = eur_val - fee_eur  # eur_val positive for sells
            cost, remainder = relieve_lot(asset, units_to_sell)
            realized = proceeds_net - cost
            rows.append({
                "date_utc": r["date_utc"], "exchange": r["exchange"], "txid": r["txid"],
                "asset": asset, "side": "sell", "qty": units_to_sell,
                "proceeds_eur": proceeds_net, "cost_eur": cost, "fees_eur": fee_eur,
                "realized_pnl_eur": realized, "short_sold_without_inventory": bool(remainder>1e-18)
            })

    pnl = pd.DataFrame(rows)
    if not pnl.empty:
        pnl["month"] = pd.to_datetime(pnl["date_utc"]).dt.tz_localize(None).dt.to_period("M").astype(str)

    realized = pnl[pnl["side"]=="sell"].copy()
    monthly = realized.groupby(["month","asset"], as_index=False).agg(
        qty_sold=("qty","sum"),
        proceeds_eur=("proceeds_eur","sum"),
        cost_eur=("cost_eur","sum"),
        fees_eur=("fees_eur","sum"),
        realized_pnl_eur=("realized_pnl_eur","sum")
    ).sort_values(["month","asset"])

    inv_rows = []
    for asset, L in lots.items():
        total_units = sum(l["units"] for l in L)
        total_cost  = sum(l["total_cost"] for l in L)
        avg_cost    = (total_cost / total_units) if total_units else 0.0
        inv_rows.append({
            "asset": asset, "closing_units": total_units, "closing_cost_eur": total_cost,
            "avg_cost_eur_per_unit": avg_cost
        })
    inventory = pd.DataFrame(inv_rows).sort_values("asset") if inv_rows else pd.DataFrame(
        columns=["asset","closing_units","closing_cost_eur","avg_cost_eur_per_unit"]
    )
    return pnl, monthly, inventory

def build_monthly_journal(monthly_pnl: pd.DataFrame, params: dict) -> pd.DataFrame:
    rows = []
    for _, r in monthly_pnl.iterrows():
        month    = r["month"]
        asset    = r["asset"]
        proceeds = float(r["proceeds_eur"] or 0.0)
        cost     = float(r["cost_eur"] or 0.0)
        fees     = float(r["fees_eur"] or 0.0)
        pnl      = float(r["realized_pnl_eur"] or 0.0)

        acc_cash = params.get("cash_eur", "101000 Cash EUR")
        acc_inv  = f'{params.get("crypto_inventory_prefix","1460 Crypto Asset ")}{asset}'
        acc_fee  = params.get("fees_expense", "611500 Trading Fees")
        acc_pnl  = params.get("realized_pnl", "701000 Trading Gains/Losses")

        # Dr Cash (proceeds)
        if abs(proceeds) > 1e-10:
            rows.append({"month": month, "account": acc_cash, "debit": max(proceeds,0.0), "credit": max(-proceeds,0.0), "asset": asset, "memo": f"Sells proceeds {asset}"})
        # Cr Inventory (cost relieved)
        if abs(cost) > 1e-10:
            rows.append({"month": month, "account": acc_inv,  "debit": 0.0, "credit": cost, "asset": asset, "memo": f"Relieve cost {asset}"})
        # Dr Fees Expense
        if abs(fees) > 1e-10:
            rows.append({"month": month, "account": acc_fee,  "debit": max(fees,0.0), "credit": max(-fees,0.0), "asset": asset, "memo": f"Trading fees {asset}"})
        # PnL
        if abs(pnl) > 1e-10:
            if pnl >= 0:
                rows.append({"month": month, "account": acc_pnl, "debit": 0.0, "credit": pnl, "asset": asset, "memo": f"Realized PnL {asset}"})
            else:
                rows.append({"month": month, "account": acc_pnl, "debit": -pnl, "credit": 0.0, "asset": asset, "memo": f"Realized PnL {asset}"})

    jl = pd.DataFrame(rows)
    _force_numeric(jl, ["debit","credit"])
    return jl

def write_csvs(pnl_detailed, monthly, inventory, jl, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coerce numerics to avoid "text numbers" in CSVs
    _force_numeric(pnl_detailed, ["proceeds_eur","cost_eur","fees_eur","realized_pnl_eur"])
    _force_numeric(monthly, ["qty_sold","proceeds_eur","cost_eur","fees_eur","realized_pnl_eur"])
    _force_numeric(inventory, ["closing_units","closing_cost_eur","avg_cost_eur_per_unit"])

    (out_dir/"realized_pnl_trades.csv").write_text(pnl_detailed.to_csv(index=False))
    (out_dir/"realized_pnl_monthly.csv").write_text(monthly.to_csv(index=False))
    (out_dir/"closing_inventory.csv").write_text(inventory.to_csv(index=False))
    (out_dir/"journal_monthly.csv").write_text(jl.to_csv(index=False))

def write_excel_report(unified: pd.DataFrame, pnl_detailed, monthly, inventory, jl, xlsx_path: Path):
    # Make date_utc timezone-naive for Excel
    if "date_utc" in unified.columns:
        unified = unified.copy()
        unified["date_utc"] = pd.to_datetime(unified["date_utc"]).dt.tz_localize(None)

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        # Parameters sheet (regenerate each run so it's always visible)
        params_df = pd.DataFrame({
            "key": ["cash_eur","cash_usd_like","crypto_inventory_prefix","fees_expense","realized_pnl"],
            "value": [PARAM_ACCOUNTS["cash_eur"], PARAM_ACCOUNTS["cash_usd_like"], PARAM_ACCOUNTS["crypto_inventory_prefix"],
                      PARAM_ACCOUNTS["fees_expense"], PARAM_ACCOUNTS["realized_pnl"]],
            "note": ["Change to your Odoo CoA","If splitting USDT separately, edit journal builder later",
                     "Prefix concatenated with asset symbol","Expense account for trading fees","Income account for realized gains/losses"],
        })
        params_df.to_excel(xw, sheet_name="Parameters", index=False)

        # Instructions
        instr = pd.DataFrame({
            "Step": [1,2,3,4,5],
            "Action": [
                "Put unified_transactions.csv next to this script (or edit INPUT_PATH).",
                "Run step2_fifo_and_journals.py",
                "Review FIFO detailed and monthly P&L sheets",
                "Check Closing Inventory balances",
                "Import Journal_Monthly to Odoo (after mapping accounts)"
            ]
        })
        instr.to_excel(xw, sheet_name="Instructions", index=False)

        # Unified input snapshot
        unified.to_excel(xw, sheet_name="Unified_Input", index=False)

        # Outputs (typed)
        _force_numeric(pnl_detailed, ["proceeds_eur","cost_eur","fees_eur","realized_pnl_eur"])
        _force_numeric(monthly, ["qty_sold","proceeds_eur","cost_eur","fees_eur","realized_pnl_eur"])
        _force_numeric(inventory, ["closing_units","closing_cost_eur","avg_cost_eur_per_unit"])
        _force_numeric(jl, ["debit","credit"])

        pnl_detailed.to_excel(xw, sheet_name="FIFO_Detailed", index=False)
        monthly.to_excel(xw, sheet_name="PnL_Monthly", index=False)
        inventory.to_excel(xw, sheet_name="Inventory_Closing", index=False)
        jl.to_excel(xw, sheet_name="Journal_Monthly", index=False)

def main():
    unified = load_unified(INPUT_PATH)
    pnl_detailed, monthly, inventory = fifo_pnl(unified)
    jl = build_monthly_journal(monthly, PARAM_ACCOUNTS)

    # Write CSVs (with numeric types enforced)
    write_csvs(pnl_detailed, monthly, inventory, jl, OUTPUT_DIR)

    # Also create/update the Excel reporting workbook automatically
    write_excel_report(unified, pnl_detailed, monthly, inventory, jl, WORKBOOK_XLS)

    print("Done.")
    print("CSV outputs  ->", OUTPUT_DIR.resolve())
    print("Excel report ->", WORKBOOK_XLS.resolve())

if __name__ == "__main__":
    main()
