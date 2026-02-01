#!/usr/bin/env python3
import sys
import re
import os

FILES_TO_UPDATE = {
    "pyproject.toml": r'version = "(.*?)"',
    "photo_quality_analyzer_core/__init__.py": r'__version__ = "(.*?)"',
}

def get_current_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
        match = re.search(FILES_TO_UPDATE["pyproject.toml"], content)
        if match:
            return match.group(1)
    return None

def bump_version(current, part):
    major, minor, patch = map(int, current.split("."))
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    return f"{major}.{minor}.{patch}"

def update_files(new_version):
    for filename, pattern in FILES_TO_UPDATE.items():
        if not os.path.exists(filename):
            print(f"⚠️ Warning: {filename} not found, skipping.")
            continue
            
        with open(filename, "r") as f:
            content = f.read()
            
        new_content = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), new_version), content)
        
        with open(filename, "w") as f:
            f.write(new_content)
        print(f"✅ Updated {filename} to {new_version}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/bump_version.py [major|minor|patch|VERSION]")
        sys.exit(1)
        
    action = sys.argv[1]
    current = get_current_version()
    
    if not current:
        print("❌ Error: Could not find current version in pyproject.toml")
        sys.exit(1)
        
    if action in ["major", "minor", "patch"]:
        new_version = bump_version(current, action)
    else:
        new_version = action # Direct override
        
    print(f"🚀 Bumping version: {current} -> {new_version}")
    update_files(new_version)
    
    # Special case: check if requirements.txt has a self-reference (common in dev setups)
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        with open("requirements.txt", "w") as f:
            for line in lines:
                # If requirements.txt has something like photo-quality-analyzer-core==0.2.0
                if "photo-quality-analyzer-core" in line and "==" in line:
                    line = re.sub(r"==.*", f"=={new_version}\n", line)
                f.write(line)
        print("✅ Checked requirements.txt for self-references.")

if __name__ == "__main__":
    main()
