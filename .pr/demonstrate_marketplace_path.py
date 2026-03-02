#!/usr/bin/env python3
"""Demonstration script for the marketplace_path feature.

This script demonstrates:
1. Loading skills from the default marketplace
2. Loading skills with marketplace_path=None (all skills)
"""

import sys


sys.path.insert(0, "openhands-sdk")

from openhands.sdk.context.skills import load_public_skills  # noqa: E402


def main():
    print("=" * 60)
    print("Marketplace Path Feature Demonstration")
    print("=" * 60)

    # 1. Load from default marketplace
    print("\n1. Loading skills from default marketplace...")
    default_skills = load_public_skills()
    print(f"   Loaded {len(default_skills)} skills from default marketplace:")
    for skill in sorted(default_skills, key=lambda s: s.name)[:5]:
        print(f"   - {skill.name}")
    if len(default_skills) > 5:
        print(f"   ... and {len(default_skills) - 5} more")

    # 2. Load all skills (no marketplace filtering)
    print("\n2. Loading ALL skills (marketplace_path=None)...")
    all_skills = load_public_skills(marketplace_path=None)
    print(f"   Loaded {len(all_skills)} skills without filtering:")
    for skill in sorted(all_skills, key=lambda s: s.name)[:5]:
        print(f"   - {skill.name}")
    if len(all_skills) > 5:
        print(f"   ... and {len(all_skills) - 5} more")

    # Verify feature works
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    if len(default_skills) > 0 and len(all_skills) >= len(default_skills):
        print("✓ Default marketplace loaded successfully")
        print("✓ All skills (no filter) loaded successfully")
        print("✓ marketplace_path parameter works correctly!")
    else:
        print("✗ Something went wrong")
        return 1
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
