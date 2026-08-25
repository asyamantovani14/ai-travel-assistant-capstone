#!/usr/bin/env python3
"""Capture and validate Atlas desktop and mobile layouts with installed Edge."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ui"
BASE_URL = "http://127.0.0.1:8000"


def capture(page, name, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(BASE_URL, wait_until="networkidle")
    page.screenshot(path=OUTPUT_DIR / f"atlas-{name}-playwright.png")
    metrics = page.evaluate(
        """() => ({
            viewportWidth: document.documentElement.clientWidth,
            contentWidth: document.documentElement.scrollWidth,
            sendButton: (() => {
                const box = document.querySelector('#sendButton').getBoundingClientRect();
                return { left: box.left, right: box.right, width: box.width };
            })(),
            title: document.title,
            imageLoaded: document.querySelector('.cover img').complete && document.querySelector('.cover img').naturalWidth > 0
        })"""
    )
    if metrics["contentWidth"] > metrics["viewportWidth"]:
        raise RuntimeError(f"Horizontal overflow in {name}: {metrics}")
    button = metrics["sendButton"]
    if button["width"] <= 0 or button["left"] < 0 or button["right"] > width:
        raise RuntimeError(f"Send button is outside the {name} viewport: {metrics}")
    if not metrics["imageLoaded"]:
        raise RuntimeError(f"Destination image failed to load in {name}")
    return metrics


def capture_conversation(page):
    page.set_viewport_size({"width": 1440, "height": 900})
    page.route(
        "**/api/chat",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "answer": "## A considered week in Sicily\n\nStart in Palermo, then travel east at a relaxed pace.",
                    "trip": {"destination": "Sicily", "duration": 7, "budget": 2400},
                    "sources": [
                        {
                            "title": "A local guide to Sicily",
                            "excerpt": "Markets, coastal rail journeys, and small-town stays.",
                            "url": "https://example.com/sicily",
                            "score": 0.91,
                        }
                    ],
                }
            ),
        ),
    )
    page.goto(BASE_URL, wait_until="networkidle")
    page.evaluate("localStorage.clear()")
    page.reload(wait_until="networkidle")
    page.get_by_text("Sicily slowly", exact=True).click()
    page.get_by_text("A considered week in Sicily", exact=True).wait_for()
    page.get_by_text("1 matched", exact=True).wait_for()
    page.get_by_text("Sicily", exact=True).last.wait_for()
    page.screenshot(path=OUTPUT_DIR / "atlas-conversation-playwright.png")
    return {"messageRendered": True, "tripUpdated": True, "sourcesUpdated": True}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    results = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        results["desktop"] = capture(page, "desktop", 1440, 900)
        results["mobile"] = capture(page, "mobile", 390, 844)
        results["conversation"] = capture_conversation(page)
        browser.close()
    if errors:
        raise RuntimeError(f"Browser errors: {errors}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
