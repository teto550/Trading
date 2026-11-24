import sys, os
sys.path.insert(0, "backtest"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import backtest as bt, egxdata
panel = bt.fetch_yfinance(bt.load_tickers())
C, H, L, V, O = bt.clean_panel(panel)
C, H, L, V, O, note = egxdata.extend_panel(C, H, L, V, O, live=True)
_f, _m, _n, _noc, lock, liquid, _fl = bt.build_features(C, H, L, V, O, 1_000_000)
r = (C / C.shift(1) - 1).where(lambda x: x.abs() <= 0.25).where(liquid)
idx = (1 + r.mean(axis=1).fillna(0)).cumprod()
sma = idx.rolling(100).mean()
on = idx > sma
print("last date", C.index[-1].date(), note)
print("idx/sma100 last 15:"); print(pd.DataFrame({"idx": idx, "sma100": sma, "gap%": (idx / sma - 1) * 100, "on": on}).tail(15).round(3).to_string())
chg = on.ne(on.shift())
runs = on[chg].index
last_sw = runs[-1]
print("current state", bool(on.iloc[-1]), "since", last_sw.date(), "sessions", int((on.index > last_sw).sum()) + 1)
s = on[on.index >= "2012-01-01"]
print("share ON 2012+: %.0f%%, 2022+: %.0f%%" % (s.mean() * 100, on[on.index >= "2022-01-01"].mean() * 100))
grp = (s != s.shift()).cumsum()
lens = s.groupby(grp).agg(["first", "size"])
off = lens[lens["first"] == False]["size"]
print("OFF streaks 2012+: n=%d median=%d p75=%d max=%d" % (len(off), off.median(), off.quantile(.75), off.max()))
for y, g in on[on.index >= "2018-01-01"].groupby(on[on.index >= "2018-01-01"].index.year):
    print(y, "on %.0f%%" % (g.mean() * 100))
# how much rise needed: solve idx_next > mean of last 99 + idx_next /100
last99 = idx.iloc[-99:].sum()
need = last99 / 99
print("index must exceed ~ %.3f vs now %.3f -> +%.1f%%" % (need, idx.iloc[-1], (need / idx.iloc[-1] - 1) * 100))
sm = (C.pipe(lambda x: x) )
