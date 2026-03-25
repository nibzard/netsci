import subprocess
import re
import sys
from pathlib import Path
import tomllib

def get_latest_version(package_name):
    try:
        result = subprocess.run(
            ["uv", "pip", "search", package_name], 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            # Try PyPI API search as fallback
            result = subprocess.run(
                ["curl", "-s", f"https://pypi.org/pypi/{package_name}/json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return data.get("info", {}).get("version", "Unknown")
        
        # Parse the output to find the latest version
        lines = result.stdout.split('\n')
        for line in lines:
            if package_name in line:
                version_match = re.search(r'\d+\.\d+\.\d+', line)
                if version_match:
                    return version_match.group(0)
        
        return "Unknown"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    pyproject_path = Path("pyproject.toml")
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject.get("project", {}).get("dependencies", [])
    updated_dependencies = []

    for dependency in dependencies:
        match = re.match(r'([a-zA-Z0-9_.-]+)([<>=~!].+)?', dependency)
        if match:
            package_name = match.group(1)
            latest_version = get_latest_version(package_name)

            if latest_version != "Unknown":
                updated_dependencies.append(f"{package_name}>={latest_version}\n")
            else:
                updated_dependencies.append(f"{dependency}\n")
        else:
            updated_dependencies.append(f"{dependency}\n")

    output_path = Path("pyproject.dependencies.new.txt")
    output_path.write_text("".join(updated_dependencies))

    print(f"Updated dependencies written to {output_path}")

if __name__ == "__main__":
    main()
