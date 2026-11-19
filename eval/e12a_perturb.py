# -*- coding: utf-8 -*-
"""E12a：对 163 节点图做结构扰动（随机删 30% 边 / 随机重连 30% 边）。

动机（计划 §1.4）：外部效度问题不必换数据集，先回答「E3 的机制排序是不是为这张图特调」。
两种扰动都只改**边**，节点与文本完全不动，因此向量臂不变，只有 PAR 的可走路径变化。

- del30：随机删除 30% 的边，其余保持；
- rewire30：随机删除 30% 的边，再按**被删边的类型分布**补同样数量的随机新边
  （不重复、不自环），保持边类型比例与总边数不变。

用法：
    python -m eval.e12a_perturb
    python -m eval.e12a_perturb --seed 20261119 --ratio 0.30

输出：
    data/dba_eval/e12a_del30.yaml
    data/dba_eval/e12a_rewire30.yaml
    eval/outputs/e12a/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

SRC = ROOT / "data/dba_eval/comprehensive_gold.yaml"


def _load() -> Dict:
    with open(SRC, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _delete(data: Dict, ratio: float, rng: random.Random) -> Dict:
    edges = list(data["edges"])
    k = max(1, round(len(edges) * ratio))
    drop = set(rng.sample(range(len(edges)), k))
    kept = [e for i, e in enumerate(edges) if i not in drop]
    return {**data, "edges": kept}, sorted(drop), [edges[i] for i in drop]


def _rewire(data: Dict, ratio: float, rng: random.Random) -> Dict:
    edges = list(data["edges"])
    k = max(1, round(len(edges) * ratio))
    drop = set(rng.sample(range(len(edges)), k))
    removed = [edges[i] for i in drop]
    kept = [e for i, e in enumerate(edges) if i not in drop]

    node_ids = [str(n["id"]) for n in data["nodes"]]
    existing = {(str(e["from"]), str(e["to"])) for e in kept}
    types = [str(e.get("type") or "scenario") for e in removed]
    rng.shuffle(types)

    added: List[Dict] = []
    guard = 0
    while len(added) < len(removed) and guard < 200000:
        guard += 1
        u, v = rng.sample(node_ids, 2)
        if (u, v) in existing:
            continue
        existing.add((u, v))
        added.append({"from": u, "to": v, "type": types[len(added)]})

    if len(added) < len(removed):
        raise RuntimeError(f"重连未完成：需要 {len(removed)} 条，只生成 {len(added)} 条")
    return {**data, "edges": kept + added}, removed, added


def _edge_types(data: Dict) -> Dict[str, int]:
    return dict(sorted(Counter(str(e.get("type")) for e in data["edges"]).items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="E12a：图结构扰动（del30 / rewire30）")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out-dir", default="data/dba_eval")
    ap.add_argument("--manifest", default="eval/outputs/e12a/manifest.json")
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rng = random.Random(args.seed)

    out_dir = ROOT / args.out_dir
    del_data, del_idx, del_edges = _delete(data, args.ratio, rng)
    rewire_data, rew_removed, rew_added = _rewire(data, args.ratio, rng)

    del_path = out_dir / "e12a_del30.yaml"
    rew_path = out_dir / "e12a_rewire30.yaml"
    _dump(del_data, del_path)
    _dump(rewire_data, rew_path)

    manifest = {
        "experiment": "E12a",
        "source": args.src,
        "seed": args.seed,
        "ratio": args.ratio,
        "orig": {"nodes": len(data["nodes"]), "edges": len(data["edges"]),
                 "edge_types": _edge_types(data)},
        "del30": {"yaml": str(del_path.relative_to(ROOT)), "edges": len(del_data["edges"]),
                  "edge_types": _edge_types(del_data),
                  "removed": [{"from": e["from"], "to": e["to"], "type": e["type"]}
                              for e in del_edges]},
        "rewire30": {"yaml": str(rew_path.relative_to(ROOT)), "edges": len(rewire_data["edges"]),
                     "edge_types": _edge_types(rewire_data),
                     "removed": [{"from": e["from"], "to": e["to"], "type": e["type"]}
                                 for e in rew_removed],
                     "added": rew_added},
    }
    mp = ROOT / args.manifest
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"原始：{len(data['nodes'])} 节点 / {len(data['edges'])} 边  {_edge_types(data)}")
    print(f"del30：{len(del_data['edges'])} 边（删 {len(del_edges)}） -> {del_path}")
    print(f"rewire30：{len(rewire_data['edges'])} 边（删 {len(rew_removed)}，补 {len(rew_added)}）"
          f" -> {rew_path}")
    print(f"[E12a] manifest wrote {mp}")


if __name__ == "__main__":
    main()
