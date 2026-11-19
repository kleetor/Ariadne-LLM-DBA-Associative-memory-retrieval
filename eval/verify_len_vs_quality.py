# -*- coding: utf-8 -*-
"""验证「故事字数 ≠ 故事质量」：统计各轮纯选中"胜者字数更长/更短/平"的分布"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    ("P vs Graphiti", "eval/outputs/story_pair_pg.json"),
    ("Naive RAG vs Graphiti", "eval/outputs/story_pair_rg.json"),
    ("P vs Naive RAG", "eval/outputs/story_pair_v3.json"),
    ("P vs Graphiti 原始事实", "eval/outputs/story_pair_p_facts.json"),
]


def main() -> None:
    for label, path in PAIRS:
        d = json.load(open(ROOT / path, encoding="utf-8"))
        longer_win = shorter_win = tie_same = 0
        examples = []
        for r in d["results"]:
            lens = r.get("story_len", {})
            if not lens:
                continue
            lv = list(lens.values())
            if len(lv) < 2:
                continue
            winner = r["winner"]
            if winner == "tie":
                tie_same += 1
                continue
            wkey = next((k for k in lens if k.lower() == winner.lower()), None)
            if wkey is None:
                continue
            wlen = lens[wkey]
            other_len = min(lv) if wlen == max(lv) else max(lv)
            if wlen > other_len:
                longer_win += 1
            else:
                shorter_win += 1
                if len(examples) < 3:
                    names = list(lens.keys())
                    olen = max(lv) if wlen == min(lv) else min(lv)
                    examples.append(f"    {r['query_id']}: {winner}（{wlen}字）胜 {[n for n in names if lens[n]==olen][0]}（{olen}字）")
        total = longer_win + shorter_win + tie_same
        print(f"[{label}] n={total}")
        print(f"  字数更长者胜: {longer_win} ({longer_win/total:.1%})")
        print(f"  字数更短者胜: {shorter_win} ({shorter_win/total:.1%})")
        print(f"  平局: {tie_same}")
        for e in examples:
            print(e)
        print()


if __name__ == "__main__":
    main()
