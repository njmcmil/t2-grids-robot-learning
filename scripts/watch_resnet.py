#!/usr/bin/env python3
"""Render a larger view of a recorded demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resnet.data import TASKS  # noqa: E402
from resnet.simulate import render_demo, side_by_side, write_mp4  # noqa: E402

OUT = ROOT / "videos" / "replay"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--demo", type=int, default=0)
    args = parser.parse_args()
    rendered = []
    for task in TASKS:
        print(f"rendering {task['name']} demo_{args.demo} at {args.size}px", flush=True)
        frames = render_demo(task, args.demo, args.size)
        path = OUT / f"watch_{task['bddl'].stem}_demo_{args.demo}.mp4"
        write_mp4(frames, path, fps=20)
        print(f"wrote {path} ({len(frames)} frames)", flush=True)
        rendered.append(frames)
    both = side_by_side(rendered[0], rendered[1])
    both_path = OUT / f"watch_both_demo_{args.demo}.mp4"
    write_mp4(both, both_path, fps=20)
    print(f"wrote {both_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
