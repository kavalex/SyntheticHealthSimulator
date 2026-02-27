#!/usr/bin/env python3
"""Version bump utility for SyntheticHealthSimulator"""

import re
import sys
from pathlib import Path


def bump_version(part: str = "patch"):

    version_file = Path("VERSION")
    current = version_file.read_text().strip()
    major, minor, patch = map(int, current.split("."))

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    new_version = f"{major}.{minor}.{patch}"

    version_file.write_text(f"{new_version}\n")

    generator = Path("generator.py")
    content = generator.read_text()
    content = re.sub(
        r'__version__ = "[\d.]+"', f'__version__ = "{new_version}"', content
    )
    content = re.sub(
        r"__version_info__ = \([\d, ]+\)",
        f"__version_info__ = ({major}, {minor}, {patch})",
        content,
    )
    generator.write_text(content)

    print(f"Version bumped: {current} → {new_version}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(part)
