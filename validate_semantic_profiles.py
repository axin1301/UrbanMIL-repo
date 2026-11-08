"""Check semantic profile coverage and flag all-zero fallback records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GROUPS = (
    "urban_form", "economic_function", "livability", "mobility",
    "socioeconomic_prior",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_json", type=Path, required=True)
    parser.add_argument("--semantic_profiles", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    profiles = json.loads(args.semantic_profiles.read_text(encoding="utf-8"))
    if not isinstance(task, list) or not isinstance(profiles, list):
        raise ValueError("Both inputs must be JSON lists")
    expected = [str(row.get("region_id", f"sample_{index:06d}"))
                for index, row in enumerate(task)]
    actual = [str(row.get("region_id")) for row in profiles]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Profile IDs do not match task row indices exactly")
    zero_ids = []
    for row in profiles:
        values = []
        for group in GROUPS:
            content = row.get(group)
            if not isinstance(content, dict) or len(content) != 6:
                raise ValueError(f"{row['region_id']}: invalid {group} group")
            values.extend(content.values())
        if len(values) != 30 or any(
            value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1)
            for value in values
        ):
            raise ValueError(f"{row['region_id']}: invalid attribute score")
        uncertainty = row.get("uncertainty")
        if (isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float))
                or not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1):
            raise ValueError(f"{row['region_id']}: invalid uncertainty")
        if all(value == 0 for value in values) and uncertainty == 0:
            zero_ids.append(row["region_id"])
    print(f"Validated {len(profiles)} profiles for {len(task)} task regions")
    print(f"All-zero profiles requiring review: {len(zero_ids)}")
    if zero_ids:
        print("IDs: " + ", ".join(zero_ids))


if __name__ == "__main__":
    main()
