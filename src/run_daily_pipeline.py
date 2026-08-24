#!/usr/bin/env python3
"""Run the travel knowledge-base refresh pipeline from the project root."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "validate_urls",
    "scrape_blogs",
    "clean_articles",
    "generate_knowledge_base",
    "auto_enrich_pipeline",
    "evaluation",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def script_path(step: str) -> Path:
    """Return the absolute script path for a named pipeline step."""
    if step not in STEPS:
        raise ValueError(f"Unknown pipeline step: {step}")
    return PROJECT_ROOT / "src" / f"{step}.py"


def select_steps(*, only: str | None = None, from_step: str | None = None) -> list[str]:
    """Select one step or a suffix of the pipeline."""
    if only:
        if only not in STEPS:
            raise ValueError(f"Unknown pipeline step: {only}")
        return [only]
    if from_step:
        if from_step not in STEPS:
            raise ValueError(f"Unknown pipeline step: {from_step}")
        return list(STEPS[STEPS.index(from_step) :])
    return list(STEPS)


def run_step(step: str, *, dry_run: bool = False) -> None:
    """Run one pipeline step and raise when it fails."""
    path = script_path(step)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {path}")

    command = [sys.executable, str(path)]
    logger.info("Running step %s", step)
    if dry_run:
        logger.info("Dry run: %s", " ".join(command))
        return

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stdout.strip():
        logger.info("%s", process.stdout.strip())
    if process.returncode:
        if process.stderr.strip():
            logger.error("%s", process.stderr.strip())
        raise subprocess.CalledProcessError(process.returncode, command)
    logger.info("Completed step %s", step)


def run_pipeline(steps: Sequence[str], *, dry_run: bool = False) -> None:
    for step in steps:
        run_step(step, dry_run=dry_run)
    logger.info("Pipeline completed successfully")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", choices=STEPS, help="Run a single step")
    selection.add_argument("--from-step", choices=STEPS, help="Resume from this step")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the selected steps without running them"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = select_steps(only=args.only, from_step=args.from_step)
    try:
        run_pipeline(steps, dry_run=args.dry_run)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.error("Pipeline stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
