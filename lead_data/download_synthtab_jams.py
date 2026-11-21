"""Download SynthTab all_jams_midi zip via Box file_id (folder page may be dead).

URL source: guitab synthtab_fetch LABEL_FILE_ID / README Full vanity.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BOX_URL = (
    "https://rochester.app.box.com/index.php"
    "?rm=box_download_shared_file"
    "&vanity_name=SynthTab-Full"
    "&file_id=f_1486897975091"
)
MIN_BYTES = 1_000_000_000


def download(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size >= MIN_BYTES:
        print(f"already present: {out} ({out.stat().st_size} bytes)")
        return out
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--http1.1",
        "-C",
        "-",
        "-o",
        str(out),
        BOX_URL,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not out.is_file() or out.stat().st_size < MIN_BYTES:
        raise SystemExit(f"download incomplete: {out}")
    print(f"ok {out} ({out.stat().st_size} bytes)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/content/all_jams_midi.zip"),
    )
    args = p.parse_args()
    download(args.out)


if __name__ == "__main__":
    main()
