"""مين مسموحله يبعت تليجرام (notify.json). الوضع focused بيقفل التقارير الكتير ويسيب المهم بس."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def source():
    if "--commands" in sys.argv:
        return "commands"
    return os.path.basename(sys.argv[0] or "")


def allowed():
    try:
        with open(os.path.join(HERE, "notify.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return True
    if cfg.get("mode", "all") != "focused":
        return True
    ok = source() in cfg.get("allow", [])
    if not ok:
        print(f"[تليجرام] اتقفل في الوضع المركّز ({source()}) - الرسالة اتسجلت في اللوج بس")
    return ok
