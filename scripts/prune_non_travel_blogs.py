#!/usr/bin/env python3
"""Remove downloaded pages that are not useful travel articles."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOG_ROOT = (PROJECT_ROOT / "data" / "blogs").resolve()
REPORT_PATH = PROJECT_ROOT / "logs" / "blog_prune_report.csv"

TRAVEL_DOMAINS = {
    "luxuryhotelinsider.com",
    "luxuryretreats.net",
    "travelynnfamily.com",
    "wildjourney.com",
    "www.cntraveler.com",
    "www.creativelena.com",
    "www.nomadicmatt.com",
}

TRAVEL_TERMS = re.compile(
    r"\b(?:"
    r"travel|traveler|traveller|traveling|travelling|trip|trips|vacation|vacations|"
    r"holiday|holidays|tourism|tourist|tourists|destination|destinations|itinerary|"
    r"hotel|hotels|resort|resorts|hostel|hostels|airbnb|accommodation|stay|stays|"
    r"airline|airlines|airport|airports|flight|flights|flying|airfare|aviation|plane|"
    r"cruise|cruises|train|trains|railway|road trip|roadtrip|rental car|luggage|"
    r"passport|visa|expat|abroad|overseas|backpack|backpacking|camping|campsite|"
    r"hike|hiking|trek|trekking|national park|theme park|disney|beach|beaches|"
    r"island|islands|safari|tour|tours|sightseeing|landmark|landmarks|things to do|"
    r"where to stay|places to visit|city guide|travel guide|weekend getaway|"
    r"solo travel|family travel|business travel|pet friendly"
    r")\b",
    re.IGNORECASE,
)


def is_domain_content_file(path: Path) -> bool:
    return path.suffix.lower() in {".html", ".htm"}


def classify(path: Path) -> tuple[bool, str]:
    relative = path.resolve().relative_to(BLOG_ROOT)
    if len(relative.parts) == 1:
        return True, "pipeline-file"

    domain = relative.parts[0].lower()
    if not is_domain_content_file(path):
        return False, "non-content-asset"
    if domain in TRAVEL_DOMAINS:
        return True, "travel-domain"

    # General-news mirrors are classified from their URL path. Reading their
    # HTML can hydrate tens of gigabytes of OneDrive placeholders and hang the
    # cleanup for hours.
    candidate_text = " ".join(relative.parts[1:])
    if TRAVEL_TERMS.search(candidate_text.replace("-", " ").replace("_", " ")):
        return True, "travel-signal"
    return False, "no-travel-signal"


def iter_blog_files():
    for path in BLOG_ROOT.rglob("*"):
        if path.is_file():
            yield path


def classify_file(path: Path):
    keep, reason = classify(path)
    return path, keep, reason, path.stat().st_size


def prune(*, apply: bool) -> Counter:
    if not BLOG_ROOT.is_dir():
        raise FileNotFoundError(f"Blog directory not found: {BLOG_ROOT}")

    stats = Counter()
    rows = []
    worker_count = min(16, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for path, keep, reason, size in executor.map(
            classify_file, iter_blog_files(), chunksize=64
        ):
            action = "keep" if keep else "delete"
            stats[f"{action}_files"] += 1
            stats[f"{action}_bytes"] += size
            stats[reason] += 1
            if not keep:
                rows.append((action, reason, size, str(path.relative_to(BLOG_ROOT))))
                if apply:
                    path.unlink()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action", "reason", "bytes", "relative_path"])
        writer.writerows(rows)

    if apply:
        directories = sorted(
            (path for path in BLOG_ROOT.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Permanently delete classified files; default is a dry run",
    )
    args = parser.parse_args()
    stats = prune(apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Keep: {stats['keep_files']:,} files ({stats['keep_bytes'] / 2**30:.2f} GiB)")
    print(f"Delete: {stats['delete_files']:,} files ({stats['delete_bytes'] / 2**30:.2f} GiB)")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
