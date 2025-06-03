import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE: Add the project root to the path so we can import from config.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROJECT_ROOT  # noqa: E402


def get_uv_exec_path() -> str:
    """Find the uv executable path using which command."""
    try:
        result = subprocess.run(
            ["which", "uv"], capture_output=True, text=True, check=True
        )
        uv_path = result.stdout.strip()
        if not uv_path:
            raise ValueError("uv executable not found in PATH")
        return uv_path
    except subprocess.CalledProcessError:
        raise ValueError("uv executable not found in PATH")


def get_config_dict() -> dict:
    """Generate the MCP server configuration dynamically."""
    uv_exec_path = get_uv_exec_path()

    config = {
        "mcpServers": {
            "buch-ai": {
                "command": uv_exec_path,
                "args": ["--directory", f"{PROJECT_ROOT}/", "run", "app/mcp/main.py"],
                "host": "127.0.0.1",
                "port": 8050,
                "timeout": 30000,
            }
        }
    }

    return config


def write_config_file() -> None:
    """Generate and write the config.json file."""
    config = get_config_dict()
    config_path = Path(__file__).parent / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Generated config.json at {config_path}")


if __name__ == "__main__":
    write_config_file()
