#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Generation script.
Runs the full pipeline with default parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add project root to path to allow imports when running as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import generate


def main():
    """Run generation with default parameters."""
    generate(
        n_people=5000,
        years=20,
        seed=42,
        boundary_coefficient=1.35,
        log_file="output/generation.log"
    )


if __name__ == "__main__":
    main()
