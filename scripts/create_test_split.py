#!/usr/bin/env python3
"""
Create a datasplits.json file with all samples in the test split.

Usage:
    python create_test_split.py <data_folder> <prefix> <output_path>

This script scans a folder for subfolders named '{prefix}_{id}' where id is a number,
and creates a datasplits.json file with all IDs in the test split.
"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create datasplits.json with all samples in test split"
    )
    parser.add_argument(
        "data_folder", type=str, help="Folder containing subfolders named {prefix}_{id}"
    )
    parser.add_argument("prefix", type=str, help="Prefix used in subfolder names")
    parser.add_argument("output_path", type=str, help="Path where datasplits.json will be saved")

    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    output_path = Path(args.output_path)
    prefix = args.prefix

    # Validate data folder
    if not data_folder.exists():
        print(f"Error: Data folder does not exist: {data_folder}")
        return

    if not data_folder.is_dir():
        print(f"Error: Data path is not a directory: {data_folder}")
        return

    # Find all subfolders matching the pattern
    pattern = re.compile(f"^{re.escape(prefix)}(\d+)$")
    test_ids = []

    print(f"Scanning folder: {data_folder}")
    print(f"Looking for pattern: {prefix}<id>")

    for subfolder in sorted(data_folder.iterdir()):
        if subfolder.is_dir():
            match = pattern.match(subfolder.name)
            if match:
                sample_id = int(match.group(1))
                test_ids.append(sample_id)

    if not test_ids:
        print(f"Warning: No subfolders found matching pattern '{prefix}<id>'")
        print("Please check the prefix and folder contents.")
        return

    # Sort IDs
    test_ids.sort()

    print(f"\nFound {len(test_ids)} samples")
    print(f"ID range: {min(test_ids)} to {max(test_ids)}")

    # Create datasplits structure
    datasplits = {"train": [], "validation": [], "test": test_ids}

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(datasplits, f, indent=2)

    print(f"\n✓ Created datasplits.json at: {output_path}")
    print(f"  Train samples:      {len(datasplits['train'])}")
    print(f"  Validation samples: {len(datasplits['validation'])}")
    print(f"  Test samples:       {len(datasplits['test'])}")


if __name__ == "__main__":
    main()
