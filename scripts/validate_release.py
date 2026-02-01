#!/usr/bin/env python3
import sys
import subprocess
import os
import re

def run_command(command, description):
    print(f"--- Running {description} ---")
    # Set PYTHONPATH to include the current directory so tests can find the package
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    result = subprocess.run(command, capture_output=True, text=True, shell=True, env=env)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(result.stdout)
        print(result.stderr)
        return False
    print(f"✅ Success: {description}")
    return True

def get_pyproject_deps():
    deps = set()
    if not os.path.exists("pyproject.toml"):
        return deps
    with open("pyproject.toml", "r") as f:
        content = f.read()
        # Find dependencies section
        match = re.search(r"dependencies = \[(.*?)\]", content, re.DOTALL)
        if match:
            dep_lines = match.group(1).split(",")
            for line in dep_lines:
                line = line.strip().strip('"').strip("'")
                if line:
                    name = re.split(r"[><=]", line)[0].strip()
                    deps.add(name.lower())
    return deps

def get_requirements_deps():
    deps = set()
    if not os.path.exists("requirements.txt"):
        return deps
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name = re.split(r"[><=]", line)[0].strip()
                deps.add(name.lower())
    return deps

def check_consistency():
    print("--- Checking Consistency ---")
    
    # 1. Dependency Consistency
    py_deps = get_pyproject_deps()
    req_deps = get_requirements_deps()
    
    if py_deps != req_deps:
        print(f"❌ Error: Dependency mismatch between pyproject.toml and requirements.txt")
        print(f"Difference: {py_deps.symmetric_difference(req_deps)}")
        return False
    
    # 2. Version Consistency
    version_pyproject = ""
    with open("pyproject.toml", "r") as f:
        for line in f:
            if line.startswith("version = "):
                version_pyproject = line.split("=")[1].strip().strip('"').strip("'")
                break
    
    version_init = ""
    if os.path.exists("photo_quality_analyzer_core/__init__.py"):
        with open("photo_quality_analyzer_core/__init__.py", "r") as f:
            for line in f:
                if line.startswith("__version__ = "):
                    version_init = line.split("=")[1].strip().strip('"').strip("'")
                    break
    
    if version_pyproject != version_init:
        print(f"❌ Error: Version mismatch!")
        print(f"pyproject.toml: {version_pyproject}")
        print(f"photo_quality_analyzer_core/__init__.py: {version_init}")
        return False

    print("✅ Success: Consistency checks passed.")
    return True

def main():
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 1. Unit Tests
    if not run_command("python3 -m unittest discover tests", "Unit Tests"):
        sys.exit(1)
        
    # 2. Consistency Checks
    if not check_consistency():
        sys.exit(1)
        
    # 3. Build Check (Dry Run)
    if not run_command("python3 -m build --dry-run", "Build Check"):
        sys.exit(1)
        
    print("\n🚀 All validations passed! Ready to release.")

if __name__ == "__main__":
    main()
