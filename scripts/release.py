#!/usr/bin/env python3
import sys
import subprocess
import os

def run(command, description=None):
    if description:
        print(f"--- {description} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error during: {description or command}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/release.py [major|minor|patch]")
        sys.exit(1)
        
    bump_type = sys.argv[1]
    
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Add script dir to sys.path for importing bump_version
    sys.path.append(script_dir)
    
    python = sys.executable
    
    # 1. Validate
    run(f"{python} scripts/validate_release.py", "Running Validations")
    
    # 2. Bump Version
    run(f"{python} scripts/bump_version.py {bump_type}", f"Bumping Version ({bump_type})")
    
    # Get new version
    from bump_version import get_current_version
    new_version = get_current_version()
    
    # 3. Commit
    run(f"git add . && git commit -m \"Rel: v{new_version}\"", "Committing Release Changes")
    
    # 4. Tag
    run(f"git tag -a v{new_version} -m \"Release v{new_version}\"", "Tagging Release")
    
    # 5. Push
    run("git push origin mainline --tags", "Pushing to GitHub (Triggering PyPI Automation)")
    
    print(f"\n🚀 v{new_version} has been released!")
    print("GitHub: Pushed to origin/mainline")
    print("PyPI: Automation triggered via tag. Check GitHub Actions tab.")

if __name__ == "__main__":
    main()
