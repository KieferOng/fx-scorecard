import os
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
import streamlit as st

# =============================
# Helpers & constants
# =============================

def _date_window(days=365):
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    return start, end

FX_TICKERS = {
    "EUR": "EURUSD=X","GBP": "GBPUSD=X","JPY": "USDJPY=X","CHF": "USDCHF=X",
    "CAD": "USDCAD=X","AUD": "AUDUSD=X","NZD": "NZDUSD=X","NOK": "USDNOK=X","SEK": "USDSEK=X",
}
VIX_TICKER = "^VIX"
EQUITY_TICKERS = {
    "USD": "^GSPC","EUR": "^STOXX50E","GBP": "^FTSE","JPY": "^N225","CHF": "^SSMI",
    "CAD": "^GSPTSE","AUD": "^AXJO","NZD": "^NZ50","NOK": "^OSEAX","SEK": "^OMXSPI",
}
COMMODITY_TICKERS = {"WTI": "CL=F","BRENT": "BZ=F","COPPER": "HG=F","DBC": "DBC"}
FRED_10Y = {
    "USD": "DGS10","GBP": "IRLTLT01GBM156N","EUR": "IRLTLT01EZM156N","JPY": "IRLTLT01JPM156N",
    "CHF": "IRLTLT01CHM156N","CAD": "IRLTLT01CAM156N","AUD": "IRLTLT01AUM156N","NZD": "IRLTLT01NZM156N",
    "NOK": "IRLTLT01NOM156N","SEK": "IRLTLT01SEM156N",
}
FRED_3M = {
    "USD": "DGS3MO","GBP": "IR3TIB01GBM156N","EUR": "IR3TIB01EZM156N","JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N","CAD": "IR3TIB01CAM156N","AUD": "IR3TIB01AUM156N","NZD": "IR3TIB01NZM156N",
    "NOK": "IR3TIB01NOM156N","SEK": "IR3TIB01SEM156N",
}
G10 = ["EUR","GBP","JPY","CHF","CAD","AUD","NZD","NOK","SEK"]
CMD_CCY = {"CAD","NOK","AUD","NZD"}
EQ_COL = {c: f"equity_{c}" for c in ["USD","EUR","GBP","JPY","CHF","CAD","AUD","NZD","NOK","SEK"]}
CMD_MAP = {"CAD": "commodities_WTI","NOK": "commodities_BRENT","AUD": "commodities_COPPER","NZD": "commodities_DBC"}

# =============================
# Downloaders
# =============================

def yf_download(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if "Adj Close" in df.columns:
        df = df["Adj Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in yfinance result.")
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.sort_index()

@st.cache_data(show_spinner=False, ttl=3600)
def yf_download_all(days=365):
    start, end = _date_window(days)
    fx_raw = yf_download(list(FX_TICKERS.values()), start, end).rename(columns={v: k for k, v in FX_TICKERS.items()}).sort_index(axis=1)
    for k in ["JPY", "CHF", "CAD", "NOK", "SEK"]:
        if k in fx_raw.columns:
            fx_raw[k] = 1.0 / fx_raw[k]
    fx = fx_raw
    vix = yf_download(VIX_TICKER, start, end).rename(columns={VIX_TICKER: "VIX"})
    eq = yf_download(list(EQUITY_TICKERS.values()), start, end).rename(columns={v: k for k, v in EQUITY_TICKERS.items()}).sort_index(axis=1)
    cmd = yf_download(list(COMMODITY_TICKERS.values()), start, end).rename(columns={v: k for k, v in COMMODITY_TICKERS.items()}).sort_index(axis=1)
    return {"fx_spot": fx, "vix": vix, "equity": eq, "commodities": cmd}

@st.cache_data(show_spinner=False, ttl=3600)
def fred_yields_download(api_key=None):
    fred = Fred(api_key=api_key)
    start, end = _date_window(365)

    def get_series(code):
        return fred.get_series(code, observation_start=start, observation_end=end)

    ylong, yshort = {}, {}
    for ccy, code in FRED_10Y.items():
        try:
            s = get_series(code); s.name = ccy; ylong[ccy] = s
        except Exception: pass
    for ccy, code in FRED_3M.items():
        try:
            s = get_series(code); s.name = ccy; yshort[ccy] = s
        except Exception: pass

    def dailyize(df):
        if df.empty: return df
        idx = pd.bdate_range(df.index.min(), df.index.max())
        return df.reindex(idx).ffill()

    return {
        "yields_10y": dailyize(pd.concat(ylong.values(), axis=1)) if ylong else pd.DataFrame(),
        "yields_3m": dailyize(pd.concat(yshort.values(), axis=1)) if yshort else pd.DataFrame()
    }

# =============================
# Panel syncing & combining
# =============================

def build_container(yf_data: dict, fred_yields: dict) -> dict:
    c = {}
    c.update(yf_data or {})
    c.update(fred_yields or {})
    return c

def synchronise_panels(container: dict, ffill=True):
    panels = {}
    for k, df in container.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            d = df.copy()
            if not isinstance(d.index, pd.DatetimeIndex):
                d.index = pd.to_datetime(d.index)
            panels[k] = d
    if not panels:
        return container
    start = max(df.index.min() for df in panels.values())
    end   = max(df.index.max() for df in panels.values())
    idx = pd.bdate_range(start, end, freq="B")
    return {k: df.reindex(idx).ffill() for k, df in panels.items()}

def combine(data_dict, sep="_"):
    frames = []
    for key, df in data_dict.items():
        df_copy = df.copy()
        df_copy.columns = [f"{key}{sep}{col}" for col in df_copy.columns]
        frames.append(df_copy)
    return pd.concat(frames, axis=1).sort_index()

# =============================
# Driver construction & scorecard
# =============================

def month_constant_from_start(s: pd.Series) -> pd.Series:
    if s.empty:
        return s

    s2 = pd.to_numeric(s, errors="coerce").astype(float)
    s2 = s2.replace(0.0, np.nan)

    months = s2.index.to_period("M")
    uniq_months = months.unique().sort_values()

    month_val = {}
    last_val = np.nan

    for m in uniq_months:
        mask = (months == m)
        sub = s2[mask].dropna()

        if not sub.empty:
            # first non-zero/non-NA in this month
            last_val = float(sub.iloc[0])

        # if sub empty, keep last_val (may still be NaN for early months)
        month_val[m] = last_val

    # map back to daily index
    out = pd.Series(index=s2.index, dtype=float)
    for idx, m in zip(s2.index, months):
        out.at[idx] = month_val.get(m, np.nan)

    return out

def build_drivers(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().sort_index()
    out = pd.DataFrame(index=data.index)

    spx = data[EQ_COL["USD"]].astype(float)
    spx_ret20d = spx.pct_change(20)
    y3m_USD = data["yields_3m_USD"].astype(float)

    for c in G10:
        fx = data[f"fx_spot_{c}"].astype(float)
        fx_logret = np.log(fx).diff()
        fx_vol60d = fx_logret.rolling(60).std() * np.sqrt(252)
        vol_safe60 = fx_vol60d.clip(lower=1e-6)
        y3m_ccy = data[f"yields_3m_{c}"].astype(float)
        y10_ccy = data[f"yields_10y_{c}"].astype(float)
        carry_spread = y3m_ccy - y3m_USD
        riskadj_carry = carry_spread / vol_safe60
        out[f"riskadjcarrychg5d_{c}"] = riskadj_carry.diff(5)
        slope_raw = (y10_ccy - y3m_ccy).diff(5)
        out[f"slopechg5d_{c}"] = month_constant_from_start(slope_raw)
        out[f"mom20d_{c}"] = fx.pct_change(20)
        eq_local = data[EQ_COL[c]].astype(float)
        out[f"eq_rel20d_{c}"] = eq_local.pct_change(20) - spx_ret20d
        if c in CMD_MAP:
            out[f"cmd20d_{c}"] = data[CMD_MAP[c]].astype(float).pct_change(20)

    vix_cols = [c for c in data.columns if "vix" in c.lower()]
    if vix_cols:
        out["vix"] = data[vix_cols[0]].astype(float)

    return out.replace([np.inf, -np.inf], np.nan).dropna(how="all")

def _zscore_cross_section(values: pd.Series):
    mu = np.nanmean(values); sd = np.nanstd(values, ddof=0)
    return (values - mu) / sd if np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=values.index)

def _winsor(x: pd.Series, lo=-3.0, hi=3.0): return x.clip(lo, hi)

def make_fx_scorecard(
    drivers_df: pd.DataFrame,
    weights_noncmd: dict,
    weights_cmd: dict,
    lookback_tag="20d",
    winsor_limits=(-3, 3),
    currencies=None
):
    last = drivers_df.sort_index().iloc[-1]
    if currencies is None:
        currencies = sorted({col.split("_")[-1] for col in drivers_df.columns if "_" in col})
    idx = pd.Index(G10, name="ccy")

    def pull(prefix):
        cols = [f"{prefix}_{c}" for c in currencies if f"{prefix}_{c}" in last.index]
        vals = pd.Series(index=[c.split("_")[-1] for c in cols], dtype=float)
        for col in cols:
            vals[col.split("_")[-1]] = float(last[col])
        return vals.reindex(idx)

    z = pd.DataFrame(index=idx)
    z["carry"]       = _winsor(_zscore_cross_section(pull("riskadjcarrychg5d")), *winsor_limits)
    z["yield_curve"] = _winsor(_zscore_cross_section(pull("slopechg5d")), *winsor_limits)
    z["momentum"]    = _winsor(_zscore_cross_section(pull(f"mom{lookback_tag}")), *winsor_limits)

    VOL_BETA = {"CHF":1.0,"JPY":0.5,"EUR":0.0,"GBP":-0.5,"CAD":-0.5,"AUD":-1.0,"NZD":-1.0,"NOK":-1.0,"SEK":-1.0}
    vix_cols = [c for c in drivers_df.columns if c.lower().startswith("vix")]
    vol_score = pd.Series(0.0, index=idx, dtype=float)
    if vix_cols:
        vix_series = pd.to_numeric(drivers_df[vix_cols[0]], errors="coerce").dropna()
        if not vix_series.empty:
            last_date = drivers_df.index[-1]; start_date = last_date - pd.DateOffset(years=5)
            vix_win = vix_series.loc[vix_series.index >= start_date]
            mu, sd = np.nanmean(vix_win.values), np.nanstd(vix_win.values, ddof=0)
            if np.isfinite(sd) and sd > 0:
                vix_z = (vix_win.iloc[-1] - mu) / sd
                for ccy in idx:
                    vol_score.loc[ccy] = VOL_BETA.get(ccy, 0.0) * vix_z
    z["volatility"]  = _winsor(vol_score, *winsor_limits)

    z["equity_rel"]  = _winsor(_zscore_cross_section(pull(f"eq_rel{lookback_tag}")), *winsor_limits)

    cmd_raw = pull(f"cmd{lookback_tag}")
    have_cmd = [c for c in currencies if c in CMD_CCY and pd.notna(cmd_raw.get(c))]
    cmd_score = pd.Series(0.0, index=idx, dtype=float)
    if len(have_cmd) >= 2:
        cmd_score.loc[have_cmd] = _zscore_cross_section(cmd_raw.loc[have_cmd])
    z["commods"] = _winsor(cmd_score, *winsor_limits)

    driver_cols = ["carry","yield_curve","momentum","volatility","equity_rel","commods"]

    w_noncmd = {k: float(v) for k, v in (weights_noncmd or {}).items()}
    w_cmd    = {k: float(v) for k, v in (weights_cmd or {}).items()}
    for w in (w_noncmd, w_cmd):
        for f in driver_cols:
            w.setdefault(f, 0.0)
    w_noncmd["commods"] = 0.0

    contrib = pd.DataFrame(index=idx, columns=driver_cols, dtype=float)
    weighted = []
    for ccy, row in z.iterrows():
        w = w_cmd if ccy in CMD_CCY else w_noncmd
        c = {f: float(row[f]) * float(w[f]) for f in driver_cols}
        contrib.loc[ccy] = pd.Series(c)
        weighted.append(np.nansum(list(c.values())))
    contrib["weighted_score"] = weighted

    contrib = contrib.sort_values("weighted_score", ascending=False)
    z = z.loc[contrib.index]

    return contrib, z

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="FX Scorecard", page_icon="💱", layout="wide")
st.title("FX Scorecard")

st.sidebar.header("Controls")
WINSOR_LIMITS = (-3.0, 3.0)
today = pd.Timestamp.today().normalize().date()
min_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=252)).date()
snapshot_date = st.sidebar.date_input(
    "Snapshot date",
    value=today,
    min_value=min_date,
    max_value=today
)

st.sidebar.subheader("Driver Weights")

def weight_inputs_group(group_label: str, factors: list[str], last_auto: str, default_each: float = 0.20):
    with st.sidebar.expander(group_label, expanded=True):
        editable = [f for f in factors if f != last_auto]
        vals = {}

        for f in editable:
            key_f = f"{group_label}-{f}"
            raw_val = st.text_input(
                f"{f} weight",
                value=f"{default_each:.3f}",
                key=key_f,
            )
            try:
                val = float(raw_val)
            except ValueError:
                val = 0.0
            vals[f] = min(max(val, 0.0), 1.0)

        manual_sum = float(np.clip(sum(vals.values()), 0.0, 1.0))
        auto_val = 1.0 - manual_sum
        if auto_val < 0:
            st.warning("Sum of entered weights exceeds 1. Reducing the last weight to 0.")
            auto_val = 0.0

        auto_key = f"{group_label}-{last_auto}"
        st.session_state[auto_key] = f"{auto_val:.4f}"

        st.text_input(
            f"{last_auto} weight",
            value=st.session_state[auto_key],
            key=auto_key,
            disabled=True,
        )

        weights_out = {**vals, last_auto: auto_val}
        total = sum(weights_out.values())
        if 0.9999 <= total <= 1.0001:
            for k in weights_out:
                weights_out[k] = weights_out[k] / total

        return weights_out

NONCMD_FACTORS = ["carry", "yield_curve", "momentum", "volatility", "equity_rel"]
noncmd_weights = weight_inputs_group(
    "Non-commods currencies (EUR/GBP/JPY/CHF/SEK)",
    ["carry", "yield_curve", "momentum", "volatility", "equity_rel"],
    last_auto="equity_rel",
    default_each=1.0/5.0
)

CMD_FACTORS = ["carry", "yield_curve", "momentum", "volatility", "equity_rel", "commods"]
cmd_weights = weight_inputs_group(
    "Commods currencies (CAD/NOK/AUD/NZD)",
    ["carry", "yield_curve", "momentum", "volatility", "equity_rel", "commods"],
    last_auto="commods",
    default_each=1.0/5.0
)

fred_api_key = st.secrets.get("FRED_API_KEY", None) or os.getenv("FRED_API_KEY")
if not fred_api_key: st.warning("⚠️ No FRED API key provided — yield data may be empty.")

run_btn = st.button("▶ Run dashboard", width='stretch')
if not run_btn:
    st.info("Set your inputs on the left, then click **Run dashboard**.")
    st.stop()

prog = st.progress(0)
try:
    yf_data = yf_download_all(); prog.progress(30)
    fred_data = fred_yields_download(api_key=fred_api_key); prog.progress(60)
    flat_panel = combine(synchronise_panels(build_container(yf_data, fred_data))); prog.progress(80)
    drivers_df = build_drivers(flat_panel); prog.progress(90)
except Exception: st.error("Data processing failed:"); st.code(traceback.format_exc()); st.stop()

snap = pd.to_datetime(snapshot_date)
drivers_asof = drivers_df.loc[:snap]
if drivers_asof.empty: st.error("No driver data available for or before this date."); st.stop()
effective_date = drivers_asof.index[-1]
if effective_date.date() < snap.date():
    st.warning(f"⚠️ Snapshot date adjusted to last available: {effective_date.date()}")

contrib_df, z_df = make_fx_scorecard(
    drivers_asof,
    weights_noncmd=noncmd_weights,
    weights_cmd=cmd_weights,
    winsor_limits=WINSOR_LIMITS
)

st.info(f"Scorecard snapshot date: **{effective_date.date()}**")

contrib_view = contrib_df.reset_index().rename(columns={"ccy":"currency"}).set_index("currency")
vmin, vmax = contrib_view["weighted_score"].min(), contrib_view["weighted_score"].max()

def color_weighted_score(val, vmin, vmax):
    if pd.isna(val): return ""
    norm = 0.5 if vmax==vmin else (val - vmin) / (vmax - vmin)
    if norm < 0.5:
        ratio = norm/0.5; r=255; g=int(255*ratio); b=int(255*ratio)
    else:
        ratio = (norm-0.5)/0.5; r=int(255*(1-ratio)); g=255; b=int(255*(1-ratio))
    return f"background-color: rgb({r},{g},{b}); color: black;"

styled_contrib = (
    contrib_view.style
      .format(precision=4)
      .map(lambda v: color_weighted_score(v, vmin, vmax), subset=["weighted_score"])
      .set_table_styles([
          {"selector":"th","props":[("font-weight","bold"),("border","2px solid black")]},
          {"selector":"td","props":[("border","1px solid black")]}
      ])
)

st.subheader("FX Scorecard")
st.dataframe(styled_contrib, width='stretch')

z_view = z_df.reset_index().rename(columns={"ccy":"currency"}).set_index("currency")
styled_z = (
    z_view.style
      .format(precision=4)
      .set_table_styles([
          {"selector":"th","props":[("font-weight","bold"),("border","2px solid black")]},
          {"selector":"td","props":[("border","1px solid black")]}
      ])
)

st.subheader("Raw driver z-scores")
st.dataframe(styled_z, width='stretch')