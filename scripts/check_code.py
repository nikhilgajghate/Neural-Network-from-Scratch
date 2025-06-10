#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return its success status and output."""
    print(f"\n=== Running {description} ===")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✅ {description} passed!")
            return True, result.stdout
        else:
            print(f"❌ {description} failed!")
            print(result.stdout)
            print(result.stderr)
            return False, result.stderr
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False, str(e)

def main() -> int:
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src directory not found!")
        return 1

    # Get all Python files in src directory
    py_files = list(src_dir.rglob("*.py"))
    if not py_files:
        print("❌ No Python files found in src directory!")
        return 1

    print(f"Found {len(py_files)} Python files to check")

    # Run Black formatting check
    black_success, _ = run_command(
        ["black", "--check", str(src_dir)],
        "Black formatting check"
    )

    # Run Ruff linting
    ruff_success, _ = run_command(
        ["ruff", "check", str(src_dir)],
        "Ruff linting"
    )

    # Run Mypy type checking
    mypy_success, _ = run_command(
        ["mypy", str(src_dir)],
        "Mypy type checking"
    )

    # Return 1 if any check failed
    return 0 if all([black_success, ruff_success, mypy_success]) else 1

if __name__ == "__main__":
    sys.exit(main()) 