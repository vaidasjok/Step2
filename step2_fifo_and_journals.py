#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.tseries.offsets import MonthEnd
from step3_split_for_import import split_unified_journals_by_asset_month as split

# ---------- CONFIG ----------
INPUT_PATH = Path("out/unified_transactions.csv")   # can be .xlsx as well
OUTPUT_DIR = Path("out")

PARAM_ACCOUNTS = {
    # 1. EUR and fiat cash
    "cash_eur": "271001 Bank",

    # 2. USD-like (USDT/USDC/BUSD…) stored on exchanges
    "cash_usd_like": "274100 Crypto USD Wallets",   # <— your custom account

    # 3. Crypto inventory – all coins grouped into a single short-term asset account
    "crypto_inventory": "262500 Cost of acquisition of crypto assets", 
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
    "exchange_eur": "274200 Exchanges EUR"
}

# ---------- FIFO DEFICIT POLICY ----------
# "raise"     -> stop immediately on first deficit (best for correctness)
# "warn"      -> continue, but report deficits (inventory will be wrong)
# "synthetic" -> auto-insert synthetic lots to bridge deficits (report will run, but must be disclosed)
DEFICIT_POLICY = "synthetic"

# Used only when DEFICIT_POLICY == "synthetic"
# "last" -> use last known unit_cost for that asset
# "avg"  -> use current weighted-average unit_cost for that asset
# "zero" -> cost=0 (NOT recommended except for debugging)
SYNTHETIC_COST_POLICY = "last"
SYNTHETIC_COST_FALLBACK_UNIT_COST = 0.0  # used if no lots exist yet
# --- Deficit dust tolerances (units) ---
DEFICIT_DUST_TOL = {
    "USDC": 1e-4,   # ignore up to 0.0001 USDC
    "USDT": 1e-4,
    "USD":  1e-4,
    "EUR":  1e-6,
}
DEFICIT_DUST_TOL_DEFAULT = 1e-12

OPENING_BALANCES_PATH = Path("out/opening_balances.csv")
# CSV columns: asset, units, eur_cost_total
# Example row: USDT,26279.9562,26279.9562   (or your EUR valuation for that opening)

# ----------------------------

def load_opening_balances(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["asset", "units", "eur_cost_total"])
    ob = pd.read_csv(path)
    ob["asset"] = ob["asset"].astype(str).str.upper().str.strip()
    ob["units"] = pd.to_numeric(ob["units"], errors="coerce").fillna(0.0)
    ob["eur_cost_total"] = pd.to_numeric(ob["eur_cost_total"], errors="coerce").fillna(0.0)
    return ob


def _force_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_posting_date(jl: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """
    Add a posting_date column based on a YYYY-MM month column.
    Example: '2023-07' -> '2023-07-31'
    """
    if jl is None or jl.empty or month_col not in jl.columns:
        return jl

    # Convert 'YYYY-MM' to first day of month
    tmp = pd.to_datetime(jl[month_col].astype(str) + "-01", errors="coerce")

    # Move to last calendar day of the month
    tmp = tmp + MonthEnd(1)

    # Store as ISO date string (Odoo likes 'YYYY-MM-DD')
    jl["posting_date"] = tmp.dt.strftime("%Y-%m-%d")

    return jl


def load_unified(path: Path) -> pd.DataFrame:
    # 1) Read file without smart date parsing
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path, low_memory=False)

    # 2) Check required columns
    exp = [
        "date_utc", "source", "event_type",
        "base_ccy", "quote_ccy", "side",
        "qty_base", "qty_quote",
        "eur_amount", "eur_fee",
        "txid", "exchange",
    ]
    missing = [c for c in exp if c not in df.columns]
    if missing:
        raise RuntimeError(f"Unified file missing columns: {missing}")

    # 3) Keep original raw date for debugging (optional, but safe)
    df["date_utc_raw"] = df["date_utc"].astype(str).str.strip()

    # 4) Single robust parse – THIS is the only place we decide what's valid
    # parsed = pd.to_datetime(df["date_utc_raw"], errors="coerce", utc=True)
    parsed = pd.to_datetime(df["date_utc_raw"], format='ISO8601', errors="coerce")

    # Show if anything is truly invalid
    bad_mask = parsed.isna()
    if bad_mask.any():
        bad = df.loc[bad_mask].copy()
        print(f">> WARNING: {bad_mask.sum()} rows have invalid date_utc; see bad_dates.csv")
        cols_show = [
            "date_utc_raw", "source", "event_type",
            "base_ccy", "quote_ccy", "qty_base", "qty_quote",
            "eur_amount", "eur_fee", "txid", "exchange",
        ]
        bad[cols_show].to_csv("bad_dates.csv", index=False)
        # OPTIONAL: if you prefer to drop truly invalid rows, uncomment:
        # parsed = parsed[~bad_mask]
        # df = df.loc[~bad_mask].copy()

    # 5) Store the parsed datetimes
    df["date_utc"] = parsed

    # 6) Normalize side + numeric fields
    df["side"] = df["side"].astype(str).str.lower()

    for c in ["qty_base", "qty_quote", "eur_amount", "eur_fee"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def fifo_pnl(
    df: pd.DataFrame,
    verbose: bool = True,
    deficit_policy: str = DEFICIT_POLICY,
    synthetic_cost_policy: str = SYNTHETIC_COST_POLICY,
    synthetic_fallback_unit_cost: float = SYNTHETIC_COST_FALLBACK_UNIT_COST,
    deficits_out_path: str = "out/deficits.csv",
):
    df = df.copy()

    if verbose:
        print("[FIFO] Using Option A: eur_amount as acquisition cost for deposits / staking / rewards.")
        print(f"[FIFO] Deficit policy = {deficit_policy!r}")

    df["event_type"] = df["event_type"].astype(str).str.lower()
    df["side"] = df["side"].astype(str).str.lower()

    # Deterministic processing order when multiple rows share the same timestamp.
    # This prevents false deficits when (for example) withdrawal and deposit have the same date_utc.
    plus_types = {"deposit", "transfer_in", "staking", "earn", "interest", "airdrop", "reward"}
    minus_types = {"withdrawal", "transfer_out"}

    rank = pd.Series(99, index=df.index)

    # Inventory increases first
    rank[df["event_type"].isin(list(plus_types))] = 10

    # Trades: buys before sells (within the same timestamp)
    rank[(df["event_type"] == "trade") & (df["side"] == "buy")] = 20
    rank[(df["event_type"] == "trade") & (df["side"] == "sell")] = 30

    # Inventory decreases last
    rank[df["event_type"].isin(list(minus_types))] = 40

    df["_event_rank"] = rank
    df = df.sort_values(["date_utc", "_event_rank", "txid"], kind="mergesort").reset_index(drop=True)
    df.drop(columns=["_event_rank"], inplace=True)


    lots_state = {}
    
    # --- OPENING BALANCES (lots injected before processing) ---
    ob = load_opening_balances(OPENING_BALANCES_PATH)
    if not ob.empty:
        for _, r0 in ob.iterrows():
            a0 = str(r0["asset"]).upper()
            u0 = float(r0["units"])
            c0 = float(r0["eur_cost_total"])
            if u0 > 0:
                lots_state.setdefault(a0, [])
                lots_state[a0].append({
                    "units": u0,
                    "total_cost": c0,
                    "unit_cost": (c0 / u0) if u0 else 0.0
                })
        if verbose:
            print(f"[FIFO] Injected opening lots from {OPENING_BALANCES_PATH}")

    rows = []

    deficits = []  # <-- collect deficit events for audit/debug

    stats = {
        "trade_buys": 0, "trade_sells": 0,
        "deposits_costed": 0, "deposits_zero_cost": 0,
        "withdrawals": 0,
        "staking_costed": 0, "staking_zero_cost": 0,
        "base_fees_relived": 0,
        "short_sells": 0,
        "deficits": 0,
        "synthetic_lots_added": 0,
    }

    def add_lot(asset, units, total_cost):
        lots_state.setdefault(asset, [])
        lots_state[asset].append({
            "units": float(units),
            "total_cost": float(total_cost),
            "unit_cost": float(total_cost)/float(units) if units else 0.0
        })

    def estimate_unit_cost(asset: str) -> float:
        lots = lots_state.get(asset, [])
        if not lots:
            return float(synthetic_fallback_unit_cost)

        if synthetic_cost_policy == "last":
            # last lot's unit cost
            return float(lots[-1].get("unit_cost", synthetic_fallback_unit_cost) or synthetic_fallback_unit_cost)

        if synthetic_cost_policy == "avg":
            total_units = sum(l["units"] for l in lots)
            total_cost = sum(l["total_cost"] for l in lots)
            if total_units > 1e-18:
                return float(total_cost / total_units)
            return float(synthetic_fallback_unit_cost)

        # "zero" or anything unknown
        return 0.0

    def relieve_lot_core(asset, units_to_relieve):
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

    CASH_LIKE_ASSETS = {"EUR", "USD", "USDT", "USDC"}

    def handle_deficit(asset: str, missing: float, ctx: dict):
        asset = str(asset).upper()
        missing = float(missing or 0.0)

        tol = DEFICIT_DUST_TOL.get(asset, DEFICIT_DUST_TOL_DEFAULT)

        # 1) Ignore dust deficits
        if missing <= tol:
            print(
                f"[FIFO][DEFICIT][IGNORED_DUST] {asset} missing {missing:g} <= tol {tol:g} | "
                f"{ctx.get('event_type')} {ctx.get('side')} | {ctx.get('date_utc')} | "
                f"txid={ctx.get('txid')} | exchange={ctx.get('exchange')}"
            )
            return

        msg = (
            f"[FIFO][DEFICIT] {asset} missing {missing:g} units | "
            f"{ctx.get('event_type')} {ctx.get('side')} | {ctx.get('date_utc')} | "
            f"txid={ctx.get('txid')} | exchange={ctx.get('exchange')}"
        )

        # 2) raise -> stop
        if deficit_policy == "raise":
            raise RuntimeError(msg)

        # 3) warn -> record + continue (inventory stays wrong)
        if deficit_policy == "warn":
            print("[FIFO][DEFICIT][WARN]", msg)
            deficits.append({**ctx, "asset": asset, "missing_units": missing, "policy": "warn"})
            stats["deficits"] += 1
            return

        # 4) synthetic -> add synthetic lot at estimated cost
        if deficit_policy == "synthetic":
            ucost = estimate_unit_cost(asset)
            eur_cost = missing * ucost
            add_lot(asset, missing, eur_cost)
            deficits.append({**ctx, "asset": asset, "missing_units": missing, "policy": "synthetic", "unit_cost": ucost})
            stats["deficits"] += 1
            stats["synthetic_lots_added"] += 1
            print(f"[FIFO][DEFICIT][SYNTHETIC] Added lot {asset} +{missing:g} @ {ucost:.8f} EUR/unit")
            return

        # 5) bridge_cash -> only for cash-like assets; otherwise stop
        if deficit_policy == "bridge_cash":
            if asset not in CASH_LIKE_ASSETS:
                raise RuntimeError(msg + " | policy=bridge_cash only allowed for cash-like assets")

            # For stablecoins, cost basis ~ 1 EUR/unit is usually acceptable as a bridge,
            # but you can also use estimate_unit_cost(asset) if you prefer consistency.
            ucost = estimate_unit_cost(asset) if asset == "EUR" else 1.0
            eur_cost = missing * ucost

            add_lot(asset, missing, eur_cost)
            deficits.append({**ctx, "asset": asset, "missing_units": missing, "policy": "bridge_cash", "unit_cost": ucost})
            stats["deficits"] += 1
            stats["synthetic_lots_added"] += 1
            print(f"[FIFO][DEFICIT][BRIDGE_CASH] Bridged {asset} +{missing:g} @ {ucost:.8f} EUR/unit")
            return

        # unknown policy
        raise RuntimeError(msg + f" | unknown deficit_policy={deficit_policy!r}")


    def relieve_lot(asset: str, units_to_relieve: float, ctx: dict) -> float:
        cost, remaining = relieve_lot_core(asset, units_to_relieve)
        if remaining > 1e-18:
            handle_deficit(asset, remaining, ctx)
            # If synthetic, we inserted a lot; relieve again to finish the intended relief
            if deficit_policy == "synthetic":
                cost2, rem2 = relieve_lot_core(asset, units_to_relieve)
                # rem2 should now be ~0; if not, surface it
                if rem2 > 1e-18:
                    handle_deficit(asset, rem2, ctx)
                return cost2
        return cost

    def add_lot_with_cost(asset, units, eur_cost, reason):
        units = float(units)
        eur_cost = float(eur_cost or 0.0)
        if units <= 0:
            return
        add_lot(asset, units, eur_cost)
        if verbose:
            unit_cost = eur_cost / units if units else 0.0
            print(f"[FIFO] ADD LOT {reason}: {asset} +{units:g} units, cost={eur_cost:.2f} EUR, unit_cost={unit_cost:.6f}")

    for _, r in df.iterrows():
        asset = str(r["base_ccy"]).upper()
        if "." in asset:
            asset = asset.split(".")[0]  # BTC.M -> BTC for FIFO only

        et = str(r["event_type"]).lower()
        side = str(r.get("side", "")).lower()

        qty_base = float(r.get("qty_base") or 0.0)
        fee_eur  = float(r.get("eur_fee") or 0.0)
        eur_val  = float(r.get("eur_amount") or 0.0)
        fee_ccy  = str(r.get("fee_ccy") or "").upper()
        if "." in fee_ccy:
            fee_ccy = fee_ccy.split(".")[0]
        fee_amt  = float(r.get("fee") or 0.0)

        ctx = {
            "date_utc": r.get("date_utc"),
            "exchange": r.get("exchange"),
            "txid": r.get("txid"),
            "source": r.get("source"),
            "event_type": et,
            "side": side,
        }

        # 1) TRADES
        if et == "trade":
            if side == "buy" and qty_base > 0:
                total_cost = -eur_val + fee_eur
                add_lot_with_cost(asset, qty_base, total_cost, "TRADE BUY")
                stats["trade_buys"] += 1

                rows.append({
                    "date_utc": r["date_utc"], "exchange": r.get("exchange"), "txid": r.get("txid"),
                    "asset": asset, "side": "buy", "qty": qty_base,
                    "proceeds_eur": 0.0, "cost_eur": total_cost, "fees_eur": fee_eur,
                    "realized_pnl_eur": 0.0
                })

            elif side == "sell" and qty_base < 0:
                units = abs(qty_base)
                proceeds_net = eur_val - fee_eur
                cost = relieve_lot(asset, units, ctx)
                realized = proceeds_net - cost

                stats["trade_sells"] += 1

                rows.append({
                    "date_utc": r["date_utc"], "exchange": r.get("exchange"), "txid": r.get("txid"),
                    "asset": asset, "side": "sell", "qty": units,
                    "proceeds_eur": proceeds_net, "cost_eur": cost, "fees_eur": fee_eur,
                    "realized_pnl_eur": realized,
                })

        # 2) DEPOSITS / TRANSFERS IN
        elif et in {"deposit", "transfer_in"} and qty_base > 0:
            if eur_val != 0:
                eur_cost = abs(eur_val)
                add_lot_with_cost(asset, qty_base, eur_cost, et.upper())
                stats["deposits_costed"] += 1
            else:
                add_lot_with_cost(asset, qty_base, 0.0, et.upper() + " (ZERO COST)")
                stats["deposits_zero_cost"] += 1
                if verbose:
                    print(f"[FIFO][INFO] {et.upper()} with no eur_amount -> ZERO cost lot: {asset} +{qty_base:g}")

        # 3) WITHDRAWALS / TRANSFERS OUT
        elif et in {"withdrawal", "transfer_out"} and qty_base < 0:
            units = abs(qty_base)
            _ = relieve_lot(asset, units, ctx)  # cost ignored; inventory relieved
            stats["withdrawals"] += 1
            if verbose:
                print(f"[FIFO] RELIEVE LOT {et.upper()}: {asset} -{units:g} units")

        # 4) STAKING / EARN / AIRDROP etc.
        elif et in {"staking", "earn", "interest", "airdrop", "reward"} and qty_base > 0:
            if eur_val != 0:
                eur_cost = abs(eur_val)
                add_lot_with_cost(asset, qty_base, eur_cost, et.upper())
                stats["staking_costed"] += 1
            else:
                add_lot_with_cost(asset, qty_base, 0.0, et.upper() + " (ZERO COST)")
                stats["staking_zero_cost"] += 1
                if verbose:
                    print(f"[FIFO][INFO] {et.upper()} with no eur_amount -> ZERO cost lot: {asset} +{qty_base:g}")

        # 5) FEE IN BASE CURRENCY
        if fee_ccy == asset and fee_amt > 0:
            fee_ctx = dict(ctx)
            fee_ctx["event_type"] = f"{et}_fee"
            _ = relieve_lot(asset, fee_amt, fee_ctx)
            stats["base_fees_relived"] += 1
            if verbose:
                print(f"[FIFO] RELIEVE LOT FEE in base: {asset} fee {fee_amt:g} units")

    # --- build outputs (same as your current code) ---
    pnl_detailed = pd.DataFrame(rows)

    if pnl_detailed.empty:
        monthly = pd.DataFrame(columns=["month", "asset", "qty_sold", "proceeds_eur",
                                        "cost_eur", "fees_eur", "realized_pnl_eur"])
        inventory = pd.DataFrame(columns=["asset", "closing_units", "closing_cost_eur", "avg_cost_eur_per_unit"])
    else:
        pnl_detailed["month"] = pd.to_datetime(pnl_detailed["date_utc"]).dt.to_period("M").astype(str)
        realized = pnl_detailed[pnl_detailed["side"] == "sell"].copy()
        monthly = realized.groupby(["month", "asset"], as_index=False).agg(
            qty_sold=("qty", "sum"),
            proceeds_eur=("proceeds_eur", "sum"),
            cost_eur=("cost_eur", "sum"),
            fees_eur=("fees_eur", "sum"),
            realized_pnl_eur=("realized_pnl_eur", "sum"),
        ).sort_values(["month", "asset"])

        inv_rows = []
        for a, lots in lots_state.items():
            total_units = sum(l["units"] for l in lots)
            total_cost = sum(l["total_cost"] for l in lots)
            avg_cost = (total_cost / total_units) if total_units else 0.0
            inv_rows.append({
                "asset": a,
                "closing_units": total_units,
                "closing_cost_eur": total_cost,
                "avg_cost_eur_per_unit": avg_cost,
            })
        inventory = pd.DataFrame(inv_rows).sort_values("asset") if inv_rows else pd.DataFrame(
            columns=["asset", "closing_units", "closing_cost_eur", "avg_cost_eur_per_unit"]
        )

    # Write deficits report if any (even in warn/synthetic)
    if deficits:
        ddf = pd.DataFrame(deficits)
        try:
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            ddf.to_csv(deficits_out_path, index=False)
            if verbose:
                print(f"[FIFO] Deficits report written to: {deficits_out_path}")
        except Exception as e:
            print("[FIFO][WARN] Could not write deficits report:", e)

    if verbose:
        print("\n[FIFO] RUN SUMMARY")
        for k, v in stats.items():
            print(f"  - {k}: {v}")
        print(f"  - assets with open lots: {len(lots_state)}")

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
        acc_exch_eur   = params.get("exchange_eur", params.get("cash_eur","101000 Cash EUR"))
        # acc_inv  = f'{params.get("crypto_inventory_prefix","1460 Crypto Asset ")}{asset}'
        acc_inv = params["crypto_inventory"]
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
            rows.append({
                "month": month, 
                "account": acc_exch_eur, 
                "debit": proceeds if proceeds > 0 else 0.0, 
                "credit": -proceeds if proceeds < 0 else 0.0,
                # "debit": max(proceeds, 0),
                # "credit": max(-proceeds, 0), 
                "asset": asset, 
                "memo": f"Sells proceeds {asset}"
            })
        # Cr Inventory (FIFO cost of units sold)
        if abs(cost) > 1e-10:
            rows.append({
                "month": month, 
                "account": acc_inv,  
                "debit": 0.0,                    
                "credit": cost,                                      
                "asset": asset, 
                "memo": f"Relieve cost {asset}"
            })
        # Fees (Dr expense)
        if abs(fees) > 1e-10:
            rows.append({
                "month": month, 
                "account": acc_fee,  
                "debit": fees if fees > 0 else 0.0, 
                "credit": -fees if fees < 0 else 0.0,
                # "debit": max(fees, 0), 
                # "credit": max(-fees, 0),
                "asset": asset, 
                "memo": f"Trading fees {asset}"
            })
        # PnL (plug) – BEFORE fees
        # We post PnL BEFORE fees here, and Trading Fees separately. 
        #   Debits  = net proceeds + fees  = gross proceeds
        #   Credits = cost + pnl_before_fees = cost + (gross - cost) = gross            
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

    # jl = pd.DataFrame(rows)
    # # Optional: round to 2 decimals for posting
    # for c in ["debit","credit"]:
    #     if c in jl.columns:
    #         jl[c] = jl[c].round(2)
    # return jl
    
    # Build DataFrame from rows
    jl = pd.DataFrame(rows)
    if jl.empty:
        return jl
    

    # # --- helper: rebalance per month (rounding plug on PnL line) ---
    # def _rebalance_one_month(group: pd.DataFrame) -> pd.DataFrame:
    #     # 1) round first
    #     group["debit"]  = group["debit"].round(2)
    #     group["credit"] = group["credit"].round(2)

    #     diff = round(group["debit"].sum() - group["credit"].sum(), 2)
    #     if abs(diff) < 0.01:
    #         # already balanced to cents
    #         return group

    #     # 2) find PnL line to plug into
    #     pnl_account = params.get("realized_pnl", "540100 Other Income")
    #     mask_pnl = group["account"] == pnl_account

    #     if mask_pnl.any():
    #         idx = group[mask_pnl].index[0]
    #     else:
    #         # fallback: if no explicit PnL line, adjust the first line
    #         idx = group.index[0]

    #     # 3) adjust PnL:
    #     #    if debits > credits  -> increase credit
    #     #    if credits > debits  -> increase debit
    #     if diff > 0:
    #         # more debit than credit → increase credit
    #         group.loc[idx, "credit"] += diff
    #     else:
    #         # more credit than debit → increase debit
    #         group.loc[idx, "debit"]  += -diff

    #     return group

    # Apply rebalance per month (you can switch to ["month","asset"] if you want per-asset balance)
    # jl = jl.groupby("month", group_keys=False).apply(_rebalance_one_month)
    
    for c in ["debit", "credit"]:
        if c in jl.columns:
            jl[c] = jl[c].round(2)

    # Debug: print imbalance (should be 0.00 if our logic is correct)
    total_debit = float(jl["debit"].sum() if "debit" in jl.columns else 0.0)
    total_credit = float(jl["credit"].sum() if "credit" in jl.columns else 0.0)
    diff = round(total_debit - total_credit, 2)
    if abs(diff) > 0.01:
        print(f"[JOURNAL_MONTHLY][WARN] debit {total_debit:.2f} vs credit {total_credit:.2f}, diff={diff:.2f}")
        
        # vj check 
        print("start check")
        # see which (month, asset) blocks are off
        grp = jl.groupby(["month", "asset"], dropna=False).agg(
            debit_sum=("debit", "sum"),
            credit_sum=("credit", "sum"),
        )
        grp["diff"] = (grp["debit_sum"] - grp["credit_sum"]).round(2)
        bad = grp[grp["diff"].abs() > 0.01]
        if not bad.empty:
            print("\n[JOURNAL_MONTHLY][DETAIL] Imbalanced month/asset groups:")
            print(
                bad.sort_values("diff", key=lambda s: s.abs(), ascending=False)
                   .head(30)
            )
        print("end check")
        # vj end check

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

    #acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
    acc_fee        = params.get("fees_expense","611500 Trading Fees")
    acc_cash       = params.get("cash_eur", "101000 Cash EUR")
    acc_exch_eur   = params.get("exchange_eur", params.get("cash_eur","101000 Cash EUR"))

    rows = []
    for _, r in df.iterrows():
        m = r["month"]; asset = str(r["asset"])
        cost = float(r.get("cost_eur", 0.0) or 0.0)
        fees = float(r.get("fees_eur", 0.0) or 0.0)
        inv_acc = params["crypto_inventory"]
        
        exchange = r["exchange"]
        
        # 1) Dr Inventory (cost)
        if cost > 0:
            rows.append({
                "month": m, 
                "account": inv_acc, 
                "debit": cost, 
                "credit": 0.0, 
                "asset": asset, 
                "memo": f"{exchange} Buy {asset} - inventory at cost"
            })
        # 2) Dr Fees Expense
        if fees > 0:
            rows.append({
                "month": m, 
                "account": acc_fee, 
                "debit": fees, 
                "credit": 0.0, 
                "asset": asset, 
                "memo": f"{exchange} Buy {asset} - fees"
            })
        # 3) Cr Cash (cost + fees)
        total_cash_out = cost + fees
        if abs(total_cash_out) > 1e-10:
            rows.append({
                "month": m, 
                "account": acc_exch_eur, 
                "debit": 0.0, 
                "credit": total_cash_out, 
                "asset": asset, 
                "memo": f"{exchange} Buy {asset} - cash outflow (cost + fees)"
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
    
    # Parse dates robustly
    parsed = pd.to_datetime(led["date_utc"], errors="coerce", utc=True)

    # DEBUG: show ledger rows whose date_utc failed to parse
    bad_mask = (led["source"] == "ledger") & parsed.isna()
    if bad_mask.any():
        print("\n[build_ledger_journals] Ledger rows with invalid date_utc:")
        print(led.loc[bad_mask, ["date_utc", "exchange", "event_type", "txid"]].head(20))
    
    led["month"] = pd.to_datetime(led["date_utc"], errors="coerce").dt.to_period("M").astype(str)
    led["asset"] = led["base_ccy"].astype(str).str.upper()
    led["etype"] = led["event_type"].astype(str).str.lower()

    acc_bank_eur   = params.get("bank_eur", "101000 Bank EUR")
    acc_exch_eur   = params.get("exchange_eur", params.get("cash_eur","101000 Cash EUR"))
    acc_exch_usdl  = params.get("exchange_usd_like", params.get("cash_usd_like","101100 Cash USD/USDT"))
    # acc_inv_prefix = params.get("crypto_inventory_prefix","1460 Crypto Asset ")
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
        
        exchange = r["exchange"]

        # --- 1. EUR deposits / withdrawals ---
        if asset == "EUR" and et == "deposit" and eur_amt > 0:
            add_row(m, acc_exch_eur, eur_amt, 0.0, asset, f"{exchange} EUR deposit to exchange")
            add_row(m, acc_clear, 0.0, eur_amt, asset, f"{exchange} EUR deposit to exchange")

        elif asset == "EUR" and et == "withdrawal" and eur_amt < 0:
            amt = -eur_amt
            add_row(m, acc_clear, amt, 0.0, asset, f"{exchange} EUR withdrawal from exchange")
            add_row(m, acc_exch_eur, 0.0, amt, asset, f"{exchange} EUR withdrawal from exchange")
            
        # --- 1b. USD-like (USD/USDT/USDC) deposits / withdrawals ---
        elif asset in {"USD", "USDT", "USDC"} and et == "deposit" and eur_amt > 0:
            # Treat as deposit into USD-like exchange wallet, valued in EUR
            add_row(m, acc_exch_usdl, eur_amt, 0.0, asset, f"{exchange} {asset} deposit to exchange")
            add_row(m, acc_clear, 0.0, eur_amt, asset, f"{exchange} {asset} deposit clearing")

        elif asset in {"USD", "USDT", "USDC"} and et == "withdrawal" and eur_amt < 0:
            amt = -eur_amt
            add_row(m, acc_clear, amt, 0.0, asset, f"{exchange} {asset} withdrawal clearing")
            add_row(m, acc_exch_usdl, 0.0, amt, asset, f"{exchange} {asset} withdrawal from exchange")

        # --- 2. Crypto deposit (increase inventory) ---
        elif et == "deposit" and asset not in {"EUR","USD","USDT","USDC"} and eur_amt > 0:
            inv_acc = params["crypto_inventory"]
            add_row(m, inv_acc, eur_amt, 0.0, asset, f"{exchange} {asset} deposit (inventory increase)")
            add_row(m, acc_clear, 0.0, eur_amt, asset, f"{exchange} Transfer clearing")

        # --- 3. Crypto withdrawal (decrease inventory) ---
        elif et == "withdrawal" and asset not in {"EUR","USD","USDT","USDC"} and eur_amt < 0:
            amt = -eur_amt
            inv_acc = params["crypto_inventory"]
            add_row(m, acc_clear, amt, 0.0, asset, f"{exchange} Transfer clearing")
            add_row(m, inv_acc, 0.0, amt, asset, f"{exchange} {asset} withdrawal (inventory decrease)")

        # --- 4. Fees (always apply) ---
        if fee_eur > 0:
            add_row(m, acc_fee, fee_eur, 0.0, asset, f"{exchange} Fee paid")
            # credit side: which account?
            if asset in {"USD","USDT","USDC"}:
                add_row(m, acc_exch_usdl, 0.0, fee_eur, asset, f"{exchange} Fee credit")
            else:
                add_row(m, acc_exch_eur, 0.0, fee_eur, asset, f"{exchange} Fee credit")

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
    print(monthly, 'cia')
    # Add posting date column for Odoo.
    jl = add_posting_date(jl, month_col="month")
    _force_numeric(jl,       ["debit", "credit"])

    if not jl.empty:
        jl["Reference"] = jl.apply(lambda r: f"CRYPTO {r.get('exchange','ALL')} {r['month']}", axis=1)
    else:
        jl["Reference"] = jl.apply(lambda r: f"CRYPTO {r.get('exchange','ALL')}", axis=1)
        
    (OUTPUT_DIR/"journal_monthly.csv").write_text(jl.to_csv(index=False))

    print("Saved in:", OUTPUT_DIR.resolve())
    
    # Additional csv files
    # --- Extra journals: minimal, separate CSVs; no Excel changes ---
    try:
        jl_buys = build_buys_journal(pnl_detailed, PARAM_ACCOUNTS)
        # Add posting date column for Odoo.
        jl_buys = add_posting_date(jl_buys, month_col="month")
        
        jl_buys["Reference"] = jl_buys.apply(
            lambda r: f"CRYPTO {r.get('exchange','ALL')} {r['month']}", axis=1
        )
    except Exception as e:
        print("WARN: buys journal failed:", e)
        jl_buys = pd.DataFrame(columns=["month","account","debit","credit","asset","memo"])
        
                
    try:
        jl_ledger = build_ledger_journals(df, PARAM_ACCOUNTS)
        # Add posting date column for Odoo.
        jl_ledger = add_posting_date(jl_ledger, month_col="month")
        
        jl_ledger["Reference"] = jl_ledger.apply(
            lambda r: f"CRYPTO {r.get('exchange','ALL')} {r['month']}", axis=1
        )
        
    except Exception as e:
        print("WARN: ledger journal failed:", e)
        jl_ledger = pd.DataFrame(columns=["month","account","debit","credit","asset","memo","posting_date"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not jl_buys.empty:
        (OUTPUT_DIR / "journal_buys_monthly.csv").write_text(jl_buys.to_csv(index=False))
    if not jl_ledger.empty:
        (OUTPUT_DIR / "journal_ledger_monthly.csv").write_text(jl_ledger.to_csv(index=False))

    # End additional csv files
    
    # Write all three journals into one file.
    
    unified_all_journals = pd.concat([jl, jl_buys, jl_ledger], ignore_index=True)
    (OUTPUT_DIR / "unified_all_journals.csv").write_text(unified_all_journals.to_csv(index=False))
    
    split(
        input_path = "out/unified_all_journals.csv",
        output_dir = "out/ready_for_import",
    )

if __name__ == "__main__":
    main()
