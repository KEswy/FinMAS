r"""
Apply the FROZEN keyword->Shenwan-industry map (reviewer 5c: ex-ante rule, not
hand-picked from outcomes) to the candidate events. Deterministic: same text ->
same industries, for every event. Also runs the leak-free name check (5b).

Reads  data/events/_candidate_events_review.csv
Writes data/events/_candidate_events_mapped.csv  (+ affected_industries column)
No CAR, no merge -- review artifact only.
"""
from __future__ import annotations
import os, sys, io, csv, re
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)

# FROZEN map: (regex over event_name) -> list of Shenwan L1 codes. Order fixed.
KEYWORD_MAP = [
    (r"LPR|降准|存款准备金|利率|MLF|逆回购|货币|流动性|再贷款", ["801780", "801790", "801180"]),
    (r"外汇存款准备金|汇率|人民币", ["801780", "801790"]),
    (r"关税|出口管制|贸易|制裁|实体清单|FDPR|对华投资", ["801080", "801880", "801110", "801890"]),
    (r"半导体|芯片|集成电路|计算芯片|AI|人工智能|华为|中芯|美光|存储", ["801080", "801750", "801770"]),
    (r"镓|锗|锑|石墨|稀土|超硬材料|两用物项", ["801050", "801030", "801740"]),
    (r"地产|房|按揭|城中村|棚改|认房", ["801180", "801710", "801780"]),
    (r"双碳|新能源|光伏|风电|购置税|新能源汽车", ["801730", "801160", "801880", "801050"]),
    (r"军工|国防", ["801740"]),
    (r"券商|资本市场|注册制|减持|退市|北京证券交易所|北交所|IPO|程序化|转融券|融券", ["801790", "801780", "801750"]),
    (r"集采|带量采购|医保|胰岛素|药品|耗材|支架", ["801150"]),
    (r"平台|反垄断|阿里|美团", ["801760", "801750"]),
    (r"化工|石油|油|煤|大宗", ["801030", "801050", "801160"]),
    (r"农业|粮食|种业", ["801010"]),
    (r"设备更新|以旧换新|制造", ["801890", "801880", "801030"]),
]

BANNED = re.compile(r"涨|跌|暴|崩|黑色|利好|利空|受益|重挫|飙|大涨|反弹|急挫|蒸发")


def assign(name: str):
    hits = []
    for pat, codes in KEYWORD_MAP:
        if re.search(pat, name):
            hits.extend(codes)
    # dedupe preserving order
    seen = set(); out = []
    for c in hits:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def main():
    src = "data/events/_candidate_events_review.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    out_rows = []
    n_banned = n_nomap = 0
    for r in rows:
        name = r["event_name"]
        banned_hit = BANNED.findall(name)
        inds = assign(name)
        if banned_hit:
            n_banned += 1
        if not inds:
            n_nomap += 1
        r["affected_industries"] = ",".join(inds)
        r["_banned_words"] = "|".join(banned_hit)
        r["_nmap"] = str(len(inds))
        out_rows.append(r)
    dst = "data/events/_candidate_events_mapped.csv"
    cols = ["event_date", "event_type", "confidence", "affected_industries",
            "_nmap", "_banned_words", "event_name", "notes"]
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"mapped {len(out_rows)} events -> {dst}")
    print(f"  leak-free name check: {n_banned} with banned words (should be 0)")
    print(f"  no-industry-match: {n_nomap} (these would be DROPPED, not force-fit)")
    # print compact table
    print(f"\n{'date':<12}{'conf':<7}{'inds':<26}{'name'}")
    for r in out_rows:
        print(f"{r['event_date']:<12}{r['confidence']:<7}{r['affected_industries']:<26}{r['event_name'][:44]}")


if __name__ == "__main__":
    main()
