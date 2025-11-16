#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np

# ---------- CONFIG ----------
INPUT_PATH = Path("unified_transactions.csv")   # can be .xlsx as well
OUTPUT_DIR = Path("out")

PARAM_ACCOUNTS = {
    # 1. EUR and fiat cash
    "cash_eur": "271001 Bank",

    # 2. USD-like (USDT/USDC/BUSD…) stored on exchanges
    "cash_usd_like": "274100 Crypto USD Wallets",   # <— your custom account

    # 3. Crypto inventory – all coins grouped into a single short-term asset account
    "crypto_inventory_prefix": "262500 Cost of acquisition of crypto assets", 
    # (prefix is still used, but because you want ONE account, it will not append asset symbols)

    # 4. Trading fees
    # No dedicated “fees expense” account in your COA → best fit:
    "fees_expense": "631200 Other Costs and Bank Taxes",

    # 5. Realized trading gains & losses
    "realized_pnl": "540100 Other Income",

    # 6. Staking / earn / interest income
    "staking_income": "540100 Other Income",

    # 7. Clearing account for internal transfers between wallets/exchanges
    "transfers_clearing": "273001 Liquidity Transfer",

    # 8. Bank EUR (for deposits/withdrawals)
    "bank_eur": "271001 Bank",

    # 9. Exchange EUR (if exchange holds EUR balances)
    "exchange_eur": "271001 Bank"
}

# ----------------------------

def _force_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_unified(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path, parse_dates=["date_utc"], low_memory=False)
    # normalize
    exp = ["date_utc","source","event_type","base_ccy","quote_ccy","side",
           "qty_base","qty_quote","eur_amount","eur_fee","txid","exchange"]
    missing = [c for c in exp if c not in df.columns]
    if missing:
        raise RuntimeError(f"Unified file missing columns: {missing}")
    df["side"] = df["side"].astype(str).str.lower()
    for c in ["qty_base","qty_quote","eur_amount","eur_fee"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fifo_pnl(trades: pd.DataFrame):
    trades = trades.copy()
    trades = trades[(trades["source"]=="trades") & (trades["event_type"]=="trade")]
    trades = trades[trades["eur_amount"].notna()]
    trades = trades.sort_values(["date_utc","txid"]).reset_index(drop=True)

    lots_state = {}
    rows = []

    def add_lot(asset, units, total_cost):
        lots_state.setdefault(asset, [])
        lots_state[asset].append({
            "units": float(units),
            "total_cost": float(total_cost),
            "unit_cost": float(total_cost)/float(units) if units else 0.0
        })

    def relieve_lot(asset, units_to_relieve):
        cost = 0.0
        remaining = float(units_to_relieve)
        lots = lots_state.get(asset, [])
        i = 0
        while remaining > 1e-18 and i < len(lots):
            lot = lots[i]
            take = min(lot["units"], remaining)
            part_cost = take * lot["unit_cost"]
            cost += part_cost
            lot["units"] -= take
            lot["total_cost"] -= part_cost
            remaining -= take
            if lot["units"] <= 1e-18:
                lots.pop(i)
            else:
                i += 1
        lots_state[asset] = lots
        return cost, remaining

    for _, r in trades.iterrows():
        asset = str(r["base_ccy"]).upper()
        qty_base = float(r["qty_base"] or 0.0)
        fee_eur = float(r["eur_fee"] or 0.0)
        eur_val = float(r["eur_amount"] or 0.0)  # quote leg value (sign follows qty_quote)
        if r["side"] == "buy" and qty_base > 0:
            total_cost = -eur_val + fee_eur  # eur_val negative on buys
            add_lot(asset, qty_base, total_cost)
            rows.append({
                "date_utc": r["date_utc"], "exchange": r["exchange"], "txid": r["txid"],
                "asset": asset, "side": "buy", "qty": qty_base,
                "proceeds_eur": 0.0, "cost_eur": total_cost, "fees_eur": fee_eur,
                "realized_pnl_eur": 0.0
            })
        elif r["side"] == "sell" and qty_base < 0:
            units_to_sell = -qty_base
            proceeds_net = eur_val - fee_eur  # eur_val positive for sells
            cost, remainder = relieve_lot(asset, units_to_sell)
            realized = proceeds_net - cost
            rows.append({
                "date_utc": r["date_utc"], "exchange": r["exchange"], "txid": r["txid"],
                "asset": asset, "side": "sell", "qty": units_to_sell,
                "proceeds_eur": proceeds_net, "cost_eur": cost, "fees_eur": fee_eur,
                "realized_pnl_eur": realized, "short_sold_without_inventory": bool(remainder>1e-18)
            })

    pnl_detailed = pd.DataFrame(rows)
    if not pnl_detailed.empty:
        pnl_detailed["month"] = pd.to_datetime(pnl_detailed["date_utc"]).dt.to_period("M").astype(str)

    realized = pnl_detailed[pnl_detailed["side"]=="sell"].copy()
    monthly = realized.groupby(["month","asset"], as_index=False).agg(
        qty_sold=("qty","sum"),
        proceeds_eur=("proceeds_eur","sum"),
        cost_eur=("cost_eur","sum"),
        fees_eur=("fees_eur","sum"),
        realized_pnl_eur=("realized_pnl_eur","sum")
    ).sort_values(["month","asset"])

    inv_rows = []
    for asset, lots in lots_state.items():
        total_units = sum(l["units"] for l in lots)
        total_cost = sum(l["total_cost"] for l in lots)
        avg_cost = (total_cost / total_units) if total_units else 0.0
        inv_rows.append({
            "asset": asset, "closing_units": total_units, "closing_cost_eur": total_cost,
            "avg_cost_eur_per_unit": avg_cost
        })
    inventory = pd.DataFrame(inv_rows).sort_values("asset") if inv_rows else pd.DataFrame(
        columns=["asset","closing_units","closing_cost_eur","avg_cost_eur_per_unit"]
    )
    return pnl_detailed, monthly, inventory

def build_monthly_journal(monthly_pnl: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Very simple journal from realized P&L perspective only (sells)
    # Debits/Credits per asset, monthly
    rows = []
    for _, r in monthly_pnl.iterrows():
        month = r["month"]
        asset = r["asset"]
        proceeds = float(r["proceeds_eur"] or 0.0)
        cost = float(r["cost_eur"] or 0.0)
        fees = float(r["fees_eur"] or 0.0)
        pnl = float(r["realized_pnl_eur"] or 0.0)
        pnl_before_fees = pnl + fees

        # Accounts
        acc_cash = params.get("cash_eur", "101000 Cash EUR")  # adjust if you split USDT
        acc_inv  = f'{params.get("crypto_inventory_prefix","1460 Crypto Asset ")}{asset}'
        acc_fee  = params.get("fees_expense", "611500 Trading Fees")
        acc_pnl  = params.get("realized_pnl", "701000 Trading Gains/Losses")

        # For sells: recognize proceeds, relieve cost, book fees and pnl
        # Minimalistic breakdown:
        # 1) Dr Cash           proceeds
        #    Cr Inventory      cost
        #    Cr Trading PnL    (proceeds - cost - fees)  [if positive, else Dr PnL]
        #    Dr Fees Expense   fees
        # We'll represent each as a separate journal line with "month" as memo.

        # Dr Cash
        if abs(proceeds) > 1e-10:
            rows.append({"month": month, "account": acc_cash, "debit": proceeds if proceeds>0 else 0.0, "credit": -proceeds if proceeds<0 else 0.0, "asset": asset, "memo": f"Sells proceeds {asset}"})
        # Cr Inventory
        if abs(cost) > 1e-10:
            rows.append({"month": month, "account": acc_inv,  "debit": 0.0,                    "credit": cost,                                      "asset": asset, "memo": f"Relieve cost {asset}"})
        # Fees (Dr expense)
        if abs(fees) > 1e-10:
            rows.append({"month": month, "account": acc_fee,  "debit": fees if fees>0 else 0.0, "credit": -fees if fees<0 else 0.0,                  "asset": asset, "memo": f"Trading fees {asset}"})
        # PnL (plug) – BEFORE fees
        # We post PnL BEFORE fees here, and Trading Fees separately.             
        if abs(pnl_before_fees) > 1e-10:
            if pnl_before_fees >= 0:
                # Profit before fees -> credit PnL
                rows.append({
                    "month": month,
                    "account": acc_pnl,
                    "debit": 0.0,
                    "credit": pnl_before_fees,
                    "asset": asset,
                    "memo": f"Realized PnL before fees {asset}"
                })
            else:
                # Loss before fees -> debit PnL
                rows.append({
                    "month": month,
                    "account": acc_pnl,
                    "debit": -pnl_before_fees,
                    "credit": 0.0,
                    "asset": asset,
                    "memo": f"Realized PnL before fees {asset}"
                })

    jl = pd.DataFrame(rows)
    # Optional: round to 2 decimals for posting
    for c in ["debit","credit"]:
        if c in jl.columns:
            jl[c] = jl[c].round(2)
    return jl

# Build journals 2025-09-01
import pandas as pd

def build_buys_journal(pnl_detailed: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Minimal buys journal from your existing FIFO detailed table.
    Expects columns: date_utc, asset, side ('buy'), cost_eur, fees_eur
    Posting:
      Dr Inventory (cost)
      Dr Fees Expense
      Cr Cash (cost)              # simple cash: use EUR cash; refine later if needed
    """
    if pnl_detailed is None or pnl_detailed.empty:
        return pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])

    df = pnl_detailed[pnl_detailed["side"].astype(str).str.lower() == "buy"].copy()
    if df.empty:
        return pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])

    # month key (keep your current behavior; harmless if tz warning appears)
    df["month"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.to_period("M").astype(str)

    acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
    acc_fee        = params.get("fees_expense","611500 Trading Fees")
    acc_cash       = params.get("cash_eur", "101000 Cash EUR")

    rows = []
    for _, r in df.iterrows():
        m = r["month"]; asset = str(r["asset"])
        cost = float(r.get("cost_eur", 0.0) or 0.0)
        fees = float(r.get("fees_eur", 0.0) or 0.0)
        inv_acc = f"{acc_inv_prefix}{asset}"
        # 1) Dr Inventory (cost)
        if cost > 0:
            rows.append({
                "month": m, 
                "account": inv_acc, 
                "debit": cost, 
                "credit": 0.0, 
                "asset": asset, 
                "memo": f"Buy {asset} - inventory at cost"
            })
        # 2) Dr Fees Expense
        if fees > 0:
            rows.append({
                "month": m, 
                "account": acc_fee, 
                "debit": fees, 
                "credit": 0.0, 
                "asset": asset, 
                "memo": f"Buy {asset} - fees"
            })
        # 3) Cr Cash (cost + fees)
        total_cash_out = cost + fees
        if abs(total_cash_out) > 1e-10:
            rows.append({
                "month": m, 
                "account": acc_cash, 
                "debit": 0.0, 
                "credit": total_cash_out, 
                "asset": asset, 
                "memo": f"Buy {asset} - cash outflow (cost + fees)"
            })

    jl = pd.DataFrame(rows)
    if jl.empty: return jl
    _force_numeric(jl, ["debit","credit"])
    jl = jl.groupby(["month","account","asset","memo"], as_index=False).agg(debit=("debit","sum"), credit=("credit","sum"))
    jl["debit"] = jl["debit"].round(2); jl["credit"] = jl["credit"].round(2)
    return jl


def build_ledger_journals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Build monthly ledger journal from deposits, withdrawals and fees.
    Requires df to contain:
       date_utc, event_type, base_ccy, eur_amount, eur_fee
    """

    led = df.copy()
    led["month"] = pd.to_datetime(led["date_utc"], errors="coerce").dt.to_period("M").astype(str)
    led["asset"] = led["base_ccy"].astype(str).str.upper()
    led["etype"] = led["event_type"].astype(str).str.lower()

    acc_bank_eur   = params.get("bank_eur", "101000 Bank EUR")
    acc_exch_eur   = params.get("exchange_eur", params.get("cash_eur","101000 Cash EUR"))
    acc_exch_usdl  = params.get("exchange_usd_like", params.get("cash_usd_like","101100 Cash USD/USDT"))
    acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
    acc_fee        = params.get("fees_expense","611500 Trading Fees")
    acc_clear      = params.get("transfers_clearing","149900 Transfers clearing")

    rows = []

    def add_row(month, account, debit, credit, asset, memo):
        rows.append({
            "month": month,
            "account": account,
            "debit": float(debit or 0.0),
            "credit": float(credit or 0.0),
            "asset": asset,
            "memo": memo
        })

    for _, r in led.iterrows():
        m = r["month"]
        asset = r["asset"]
        et = r["etype"]
        eur_amt = float(r["eur_amount"]) if pd.notna(r["eur_amount"]) else 0.0
        fee_eur = float(r["eur_fee"]) if pd.notna(r["eur_fee"]) else 0.0

        # --- 1. EUR deposits / withdrawals ---
        if asset == "EUR" and et == "deposit" and eur_amt > 0:
            add_row(m, acc_exch_eur, eur_amt, 0.0, asset, "EUR deposit to exchange")
            add_row(m, acc_bank_eur, 0.0, eur_amt, asset, "EUR deposit to exchange")

        elif asset == "EUR" and et == "withdrawal" and eur_amt < 0:
            amt = -eur_amt
            add_row(m, acc_bank_eur, amt, 0.0, asset, "EUR withdrawal from exchange")
            add_row(m, acc_exch_eur, 0.0, amt, asset, "EUR withdrawal from exchange")

        # --- 2. Crypto deposit (increase inventory) ---
        elif et == "deposit" and asset not in {"EUR","USD","USDT","USDC"} and eur_amt > 0:
            inv_acc = f"{acc_inv_prefix}{asset}"
            add_row(m, inv_acc, eur_amt, 0.0, asset, f"{asset} deposit (inventory increase)")
            add_row(m, acc_clear, 0.0, eur_amt, asset, "Transfer clearing")

        # --- 3. Crypto withdrawal (decrease inventory) ---
        elif et == "withdrawal" and asset not in {"EUR","USD","USDT","USDC"} and eur_amt < 0:
            amt = -eur_amt
            inv_acc = f"{acc_inv_prefix}{asset}"
            add_row(m, acc_clear, amt, 0.0, asset, "Transfer clearing")
            add_row(m, inv_acc, 0.0, amt, asset, f"{asset} withdrawal (inventory decrease)")

        # --- 4. Fees (always apply) ---
        if fee_eur > 0:
            add_row(m, acc_fee, fee_eur, 0.0, asset, "Fee paid")
            # credit side: which account?
            if asset in {"USD","USDT","USDC"}:
                add_row(m, acc_exch_usdl, 0.0, fee_eur, asset, "Fee credit")
            else:
                add_row(m, acc_exch_eur, 0.0, fee_eur, asset, "Fee credit")

    jl = pd.DataFrame(rows)
    if jl.empty:
        return jl

    _force_numeric(jl, ["debit","credit"])
    jl = jl.groupby(["month","account","asset","memo"], as_index=False).agg(
        debit=("debit","sum"), credit=("credit","sum")
    )

    jl["debit"] = jl["debit"].round(2)
    jl["credit"] = jl["credit"].round(2)
    return jl


# End building journals 2025-09-01

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_unified(INPUT_PATH)

    pnl_detailed, monthly, inventory = fifo_pnl(df)

    # Save CSV outputs
    
    # Make sure numeric columns are real numbers (not text)
    _force_numeric(pnl_detailed,   ["qty", "proceeds_eur", "cost_eur", "fees_eur", "realized_pnl_eur"])
    _force_numeric(monthly,  ["qty_sold", "proceeds_eur", "cost_eur", "fees_eur", "realized_pnl_eur"])
    _force_numeric(inventory,["closing_units", "closing_cost_eur", "avg_cost_eur_per_unit"])
   

    (OUTPUT_DIR/"realized_pnl_trades.csv").write_text(pnl_detailed.to_csv(index=False))
    (OUTPUT_DIR/"realized_pnl_monthly.csv").write_text(monthly.to_csv(index=False))
    (OUTPUT_DIR/"closing_inventory.csv").write_text(inventory.to_csv(index=False))

    # Build a simple monthly journal (sells only). Extend later for buys, ledger, deposits/withdrawals.
    jl = build_monthly_journal(monthly, PARAM_ACCOUNTS)
    _force_numeric(jl,       ["debit", "credit"])
    (OUTPUT_DIR/"journal_monthly.csv").write_text(jl.to_csv(index=False))

    print("Saved in:", OUTPUT_DIR.resolve())
    
    # Additional csv files
    # --- Extra journals: minimal, separate CSVs; no Excel changes ---
    try:
        jl_buys = build_buys_journal(pnl_detailed, PARAM_ACCOUNTS)
    except Exception as e:
        print("WARN: buys journal failed:", e)
        jl_buys = pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])

    try:
        jl_ledger = build_ledger_journals(df, PARAM_ACCOUNTS)
    except Exception as e:
        print("WARN: ledger journal failed:", e)
        jl_ledger = pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not jl_buys.empty:
        (OUTPUT_DIR / "journal_buys_monthly.csv").write_text(jl_buys.to_csv(index=False))
    if not jl_ledger.empty:
        (OUTPUT_DIR / "journal_ledger_monthly.csv").write_text(jl_ledger.to_csv(index=False))

    # End additional csv files

if __name__ == "__main__":
    main()
