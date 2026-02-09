#!/usr/bin/env python3
"""
Extract changelog section for a specific version from CHANGELOG.md

Usage:
    python scripts/extract_changelog.py 0.7.0
"""

import sys
import re
from pathlib import Path


def extract_changelog_for_version(version: str) -> str:
    """
    Extract the changelog content for a specific version.
    
    Args:
        version: Version number (e.g., "0.7.0")
        
    Returns:
        The changelog content for that version, or empty string if not found.
    """
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    
    if not changelog_path.exists():
        print(f"Error: CHANGELOG.md not found at {changelog_path}", file=sys.stderr)
        sys.exit(1)
    
    content = changelog_path.read_text()
    
    # Pattern to match version section: from [version] to next ## or end of file
    # We want everything AFTER the version header line
    pattern = rf"## \[{re.escape(version)}\][^\n]*\n(.*?)(?=\n## \[|\Z)"
    
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"Error: Version {version} not found in CHANGELOG.md", file=sys.stderr)
        sys.exit(1)
    
    changelog = match.group(1).strip()
    
    return changelog


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_changelog.py <version>", file=sys.stderr)
        print("Example: python extract_changelog.py 0.7.0", file=sys.stderr)
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Remove 'v' prefix if present
    if version.startswith('v'):
        version = version[1:]
    
    changelog = extract_changelog_for_version(version)
    
    # Print to stdout (can be redirected to file)
    print(changelog)


if __name__ == "__main__":
    main()
