#!/usr/bin/env python3
"""
run_daily_pipeline.py

Orchestrates the full blog pipeline:
1) validate URLs
2) scrape blogs
3) clean articles
4) generate knowledge base
5) auto-enrich pipeline
6) evaluate metrics
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# List of scripts to run, relative to project root
SCRIPTS = [
    "src/validate_urls.py",
    "src/scrape_blogs.py",
    "src/clean_articles.py",
    "src/generate_knowledge_base.py",
    "src/auto_enrich_pipeline.py",
    "src/evaluation.py"
]

def run_step(path):
    """Invoke a python script and exit on failure."""
    full_path = os.path.abspath(path)
    logging.info(f"▶️ Running {full_path}")
    proc = subprocess.run(
        [sys.executable, full_path],
        capture_output=True,
        text=True
    )
    if proc.returncode != 0:
        logging.error(f"❌ {path} failed (exit {proc.returncode})")
        logging.error(proc.stdout)
        logging.error(proc.stderr)
        sys.exit(proc.returncode)
    logging.info(f"✅ {path} completed successfully")
    # Optionally log stdout
    if proc.stdout.strip():
        logging.info(proc.stdout.strip())

def main():
    # Change cwd to project root (this file's parent)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for script in SCRIPTS:
        run_step(script)
    logging.info("🎉 All pipeline steps finished without errors.")

if __name__ == "__main__":
    main()
