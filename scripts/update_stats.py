#!/usr/bin/env python3
"""Update stats.json after a challenge evaluation run."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: update_stats.py <challenge_dir> <exit_code>")
        sys.exit(1)

    challenge_dir = sys.argv[1]
    exit_code = int(sys.argv[2])

    stats_path = Path("stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {
        "streak": 0, "totalChallenges": 0, "solved": 0, "history": []
    }

    # Load metadata
    meta_path = Path(challenge_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    slug = Path(challenge_dir).name
    passed = exit_code == 0

    # Update or add history entry
    entry = next((h for h in stats["history"] if h["slug"] == slug), None)
    if entry:
        entry["submitted"] = True
        entry["testsPassed"] = passed
    else:
        stats["history"].append({
            "date": meta.get("date", slug[:10]),
            "slug": slug,
            "title": meta.get("title", slug),
            "category": meta.get("category", "Coding"),
            "difficulty": meta.get("difficulty", "medium"),
            "submitted": True,
            "testsPassed": passed,
        })

    # Sort history by date descending
    stats["history"].sort(key=lambda h: h["date"], reverse=True)

    # Recalculate stats
    stats["totalChallenges"] = len(stats["history"])
    stats["solved"] = sum(1 for h in stats["history"] if h.get("testsPassed"))

    # Calculate streak (consecutive passed from most recent)
    streak = 0
    for h in stats["history"]:
        if h.get("testsPassed"):
            streak += 1
        elif h.get("submitted"):
            break
    stats["streak"] = streak

    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Stats updated: {slug} {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
