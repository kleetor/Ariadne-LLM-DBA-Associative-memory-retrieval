# SPDX-License-Identifier: AGPL-3.0-only

"""GPT Image 生图脚本（文生图 / 图生图）。

API 文档要点（https://api-gpt-ge.apifox.cn）：

- 文生图：POST {base}/images/generations  JSON
  prompt ≤ 1000 字符（文档另注 4000）；model ∈
  gpt-image-2.5-flare / gpt-image-2.5-sunburst / gpt-image-2 / gpt-image-1.5
  size：gpt-image-2+ 任意边 ≤3840 且为 16 的倍数；gpt-image-1.5 仅
  1024x1024 / 1024x1536 / 1536x1024；gpt-image-2.5 仅 1k
  output_format：png / jpeg / webp；background：transparent / opaque / auto
  返回 data[].b64_json

- 图生图：POST {base}/images/edits  multipart/form-data
  image（≤4MB PNG/JPG/WebP，最多 4 张）、prompt；可选 mask（同尺寸 PNG）
  返回 data[].url 或 data[].b64_json

凭据（按优先级）：
  IMAGE_API_KEY  → JUDGE_API_KEY → OPENAI_API_KEY
  IMAGE_API_BASE → JUDGE_API_BASE → https://api.gpt.ge/v1
  IMAGE_MODEL    → gpt-image-2

用法示例：
  python imagegen/generate.py text --prompt "..." --size 1536x1024 --quality high
  python imagegen/generate.py text --scene hero
  python imagegen/generate.py edit --image assets/generated/hero.png --prompt "改成夜景"
  python imagegen/generate.py list-scenes
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
SCENES_PATH = Path(__file__).resolve().parent / "scenes.json"
DEFAULT_OUT_DIR = ROOT / "assets" / "generated"

# 各模型允许的尺寸：None 表示「任意边 ≤3840 且为 16 的倍数」
_SIZE_RULES = {
    "gpt-image-2.5-flare": ("1024x1024", "1k"),
    "gpt-image-2.5-sunburst": ("1024x1024", "1k"),
    "gpt-image-1.5": ("1024x1024", "1024x1536", "1536x1024"),
    "gpt-image-2": None,
}

# 上游饱和 / 网关抖动：文档里明确会出现「负载已饱和」，值得自动退避重试
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _post(url: str, *, retries: int = 3, **kwargs):
    """带退避重试的 POST（只对 429/5xx 与网络错误重试）。"""
    delay = 3
    resp = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, **kwargs)
        except requests.RequestException as exc:
            if attempt >= retries:
                raise
            print(f"网络错误（{exc}），{delay}s 后重试 {attempt}/{retries}")
        else:
            if resp.status_code not in _RETRY_STATUS or attempt >= retries:
                return resp
            print(f"HTTP {resp.status_code}（上游繁忙），{delay}s 后重试 {attempt}/{retries}")
        time.sleep(delay)
        delay *= 2
    return resp


def _load_env(path: Path = ENV_PATH) -> None:
    """把项目根目录 .env 读进环境变量（不覆盖已存在的值）。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _pick(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _config() -> tuple[str, str, str]:
    _load_env()
    base = _pick("IMAGE_API_BASE", "JUDGE_API_BASE", default="https://api.gpt.ge/v1").rstrip("/")
    key = _pick("IMAGE_API_KEY", "JUDGE_API_KEY", "OPENAI_API_KEY")
    model = _pick("IMAGE_MODEL", default="gpt-image-2")
    if not key:
        sys.exit("缺少 API Key：请在 .env 设置 IMAGE_API_KEY（或 JUDGE_API_KEY / OPENAI_API_KEY）")
    return base, key, model


def _resolve_size(model: str, size: str) -> str:
    """校验尺寸；模型不支持时**自动回落**到它支持的尺寸（渠道受限时很常见）。

    gpt-image-2 系列：任意边 ≤3840 且为 16 的倍数。
    gpt-image-1.5：1024x1024 / 1024x1536 / 1536x1024。
    gpt-image-2.5：当前只开放 1k（1024x1024）。
    """
    want = (size or "auto").strip().lower()
    rules = _SIZE_RULES.get(model)
    if rules is None:
        if want == "auto":
            return "auto"
        parts = want.split("x")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            sys.exit(f"尺寸格式应为 宽x高（如 1536x1024），收到 {size!r}")
        w, h = int(parts[0]), int(parts[1])
        if not (1 <= w <= 3840 and 1 <= h <= 3840):
            sys.exit("gpt-image-2 系列要求任意边 ≤3840")
        if w % 16 or h % 16:
            sys.exit("gpt-image-2 系列要求宽高都是 16 的倍数")
        return f"{w}x{h}"

    if want in rules:
        return want
    fallback = "1024x1024" if "1024x1024" in rules else rules[0]
    if want != "auto":
        print(f"提示: {model} 不支持 {want}，改用 {fallback}")
    return fallback


def _slug(text: str, limit: int = 40) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:limit]
    return ascii_part or "image"


def _save_b64(data: str, out_dir: Path, stem: str, index: int, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"jpeg": "jpg"}.get(fmt, fmt)
    name = f"{stem}-{int(time.time())}" + (f"-{index}" if index else "") + f".{suffix}"
    path = out_dir / name
    path.write_bytes(base64.b64decode(data))
    return path


def _download(url: str, out_dir: Path, stem: str, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    fmt = "png" if "png" in ctype else "jpg" if "jpeg" in ctype else "webp"
    name = f"{stem}-{int(time.time())}" + (f"-{index}" if index else "") + f".{fmt}"
    path = out_dir / name
    path.write_bytes(resp.content)
    return path


def _report(payload: dict, saved: list[Path]) -> None:
    usage = payload.get("usage") or {}
    print("已保存:")
    for path in saved:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    if usage:
        print(
            "用量: input={} output={} total={}".format(
                usage.get("input_tokens", "-"),
                usage.get("output_tokens", "-"),
                usage.get("total_tokens", "-"),
            )
        )


def generate_text(args) -> int:
    base, key, model = _config()
    prompt = args.prompt
    if args.scene:
        scenes = _load_scenes()
        if args.scene not in scenes:
            sys.exit(f"未知场景 {args.scene!r}，可选: {', '.join(scenes)}")
        scene = scenes[args.scene]
        prompt = scene["prompt"]
        # 场景里的默认值可被命令行覆盖（命令行优先）
        args.size = args.size or scene.get("size", "auto")
        args.quality = args.quality or scene.get("quality", "high")
        args.output_format = args.output_format or scene.get("output_format", "png")
        args.background = args.background or scene.get("background", "auto")

    if not prompt:
        sys.exit("必须提供 --prompt 或 --scene")
    if len(prompt) > 4000:
        sys.exit(f"prompt 过长（{len(prompt)} 字符，上限 4000）")

    size = _resolve_size(model, args.size or "auto")
    payload = {
        "model": model,
        "prompt": prompt,
        "n": max(1, min(int(args.n), 10)),
        "size": size,
        "quality": args.quality or "high",
        "output_format": args.output_format or "png",
        "background": args.background or "auto",
        "moderation": args.moderation or "auto",
    }

    print(f"请求 {base}/images/generations  model={model} size={size} "
          f"quality={payload['quality']} n={payload['n']}")
    t0 = time.time()
    resp = _post(
        f"{base}/images/generations",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    items = data.get("data") or []
    if not items:
        sys.exit(f"响应没有图片数据: {json.dumps(data, ensure_ascii=False)[:400]}")

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    stem = args.name or _slug(args.scene or prompt)
    saved = []
    for i, item in enumerate(items, 1):
        if item.get("b64_json"):
            saved.append(_save_b64(item["b64_json"], out_dir, stem, i, payload["output_format"]))
        elif item.get("url"):
            saved.append(_download(item["url"], out_dir, stem, i))
    print(f"完成，用时 {time.time() - t0:.1f}s")
    _report(data, saved)
    return 0


def generate_edit(args) -> int:
    base, key, model = _config()
    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"源图不存在: {image_path}")
    if image_path.stat().st_size > 4 * 1024 * 1024:
        sys.exit("源图超过 4MB，请先压缩")
    if not args.prompt:
        sys.exit("必须提供 --prompt")

    form = {
        "model": model,
        "prompt": args.prompt,
        "n": str(max(1, min(int(args.n), 10))),
        "size": _resolve_size(model, args.size or "auto"),
    }
    files = [("image", (image_path.name, image_path.read_bytes(), "image/png"))]
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.is_file():
            sys.exit(f"遮罩不存在: {mask_path}")
        files.append(("mask", (mask_path.name, mask_path.read_bytes(), "image/png")))

    print(f"请求 {base}/images/edits  model={model} image={image_path.name}")
    t0 = time.time()
    resp = _post(
        f"{base}/images/edits",
        headers={"Authorization": f"Bearer {key}"},
        data=form,
        files=files,
        timeout=600,
    )
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    items = data.get("data") or []
    if not items:
        sys.exit(f"响应没有图片数据: {json.dumps(data, ensure_ascii=False)[:400]}")

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    stem = args.name or _slug(image_path.stem + "-edit")
    saved = []
    for i, item in enumerate(items, 1):
        if item.get("b64_json"):
            saved.append(_save_b64(item["b64_json"], out_dir, stem, i, "png"))
        elif item.get("url"):
            saved.append(_download(item["url"], out_dir, stem, i))
    print(f"完成，用时 {time.time() - t0:.1f}s")
    _report(data, saved)
    return 0


def _load_scenes() -> dict:
    if not SCENES_PATH.is_file():
        return {}
    return json.loads(SCENES_PATH.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT Image 生图工具（文生图 / 图生图）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="文生图")
    p_text.add_argument("--prompt", default="")
    p_text.add_argument("--scene", default="", help="从 scenes.json 取预设提示词")
    p_text.add_argument("--size", default="")
    p_text.add_argument("--quality", default="", choices=["", "auto", "high", "medium", "low"])
    p_text.add_argument("--n", default=1)
    p_text.add_argument("--output-format", dest="output_format", default="",
                        choices=["", "png", "jpeg", "webp"])
    p_text.add_argument("--background", default="", choices=["", "auto", "transparent", "opaque"])
    p_text.add_argument("--moderation", default="", choices=["", "auto", "low"])
    p_text.add_argument("--out-dir", dest="out_dir", default="")
    p_text.add_argument("--name", default="")
    p_text.set_defaults(func=generate_text)

    p_edit = sub.add_parser("edit", help="图生图（multipart）")
    p_edit.add_argument("--image", required=True)
    p_edit.add_argument("--prompt", required=True)
    p_edit.add_argument("--mask", default="")
    p_edit.add_argument("--size", default="")
    p_edit.add_argument("--n", default=1)
    p_edit.add_argument("--out-dir", dest="out_dir", default="")
    p_edit.add_argument("--name", default="")
    p_edit.set_defaults(func=generate_edit)

    p_list = sub.add_parser("list-scenes", help="列出 scenes.json 里的预设场景")
    p_list.set_defaults(func=lambda a: print("\n".join(_load_scenes()) or "（scenes.json 为空）") or 0)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
