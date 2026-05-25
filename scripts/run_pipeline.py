"""End-to-end pipeline runner for one jurisdiction × one indicator.

Usage:
    python scripts/run_pipeline.py --jurisdiction SG --indicator 6.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lexora.collect.profile_loader import load_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jurisdiction", required=True, help="ISO code, e.g. SG")
    parser.add_argument("--indicator", required=True, help="RDTII indicator id, e.g. 6.1")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/jurisdictions"),
    )
    args = parser.parse_args()

    profile = load_profile(args.config_dir / f"{args.jurisdiction.lower()}.yaml")
    print(f"Loaded profile: {profile.jurisdiction} ({profile.iso_code})")
    print(f"Indicator: {args.indicator}")
    print(f"Portals: {len(profile.portals)}")
    # TODO: wire collect → extract → structure → classify → cite → export
    print("Pipeline stages not yet implemented — see TODOs in src/lexora/*.")


if __name__ == "__main__":
    main()
