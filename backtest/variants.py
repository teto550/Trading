#!/usr/bin/env python3
"""مقارنة سريعة لأوضاع "أسهم بس" وصفقات أقصر، على نفس الإعداد الحي (تذبذب منخفض) وبالتكلفة الحقيقية.
الحكم على 2012-2021 (تطوير)، و2022+ للتأكيد."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import swing as sw, swing_core as sc, backtest as bt
from types import SimpleNamespace
a = SimpleNamespace(source="yfinance", min_turnover=1_000_000)
C, H, L, V, O, liquid, lock = sw.load_data(a)
live = json.load(open(os.path.join(bt.RES_DIR, "swing_live.json")))
base = dict(sc.DEFAULT, **live["params"]); base.pop("cash_rate", None)
S = sw.make_scores(live.get("scores", "lowvol"), C, liquid)
D = sc.prepare(C, H, L, V, O, S, liquid, lock)
T = len(C); first = int(np.where(np.isfinite(S).sum(axis=1) >= 30)[0][0])
V_ = [("الحالي: كاش والسوق هابط", {}),
      ("أسهم دايمًا (من غير فلتر سوق)", {"regime": 0}),
      ("دفاعي: والسوق هابط أهدى 3% بس، 4 مراكز", {"def_mode": True, "def_q": 0.97, "def_pos": 4}),
      ("دفاعي: والسوق هابط أهدى 5%، 8 مراكز", {"def_mode": True, "def_q": 0.95, "def_pos": 8}),
      ("دفاعي 3% / 4 مراكز + وقف 3.5", {"def_mode": True, "def_q": 0.97, "def_pos": 4, "trail_k": 3.5})]
for mh in (3, 5, 10, 20):
    V_.append((f"صفقات قصيرة: أقصى {mh} جلسات", {"max_hold": mh, "min_hold": min(2, mh), "q_out": 0.5}))
    V_.append((f"صفقات قصيرة {mh} جلسات + دفاعي", {"max_hold": mh, "min_hold": min(2, mh), "q_out": 0.5, "def_mode": True}))
out = ["| الوضع | الفترة " + sw.HEAD[0], "|---|---" + sw.HEAD[1]]
for lab, ch in V_:
    p = dict(base, **ch)
    e, x, t = sc.simulate(D, p, first, T)
    for per, rng in (("2012-2021", sw.DEV), ("2022+", sw.TEST)):
        out.append(f"| {lab} | {per} " + sw.fmt(sc.metrics(D, e, x, t, *rng)))
txt = "\n".join(out)
open(os.path.join(bt.RES_DIR, "report_variants_latest.md"), "w", encoding="utf-8").write(txt)
print(txt)
