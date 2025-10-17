import requests
from packaging import version

from .terminal_style import *

def check_for_update(current_version: str, repo: str) -> None:
    """
    Check if a newer version exists on GitHub.

    :param current_version: The version string of the installed program (e.g. "1.2.3").
    :param repo: The GitHub repo in 'owner/name' format.
    """
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        latest_release = response.json()["tag_name"].lstrip("v")
    except Exception as e:
        print_error(f"Update check failed: {e}")
        return

    if version.parse(latest_release) > version.parse(current_version):
        print_warning(f"New version available: {latest_release} (you have {current_version})")