#!/usr/bin/env python3
import requests
import subprocess
import getpass

BASEURL = "https://fallingstar-data.com/forcedphot/api-token-auth/"

# Prompt for username and password
username = input("ATLAS username: ")
password = getpass.getpass("ATLAS password: ")

# Request token
resp = requests.post(BASEURL, data={"username": username, "password": password})
if resp.status_code != 200:
    print(f"Error {resp.status_code}: {resp.text}")
    exit(1)

token = resp.json()["token"]
print(f"✅ Got token: {token}")

# Save token to fish config
fish_config = "~/.config/fish/config.fish"
cmd = f'set -x ATLAS_TOKEN {token}'

# Append to fish config
subprocess.run(
    f'echo "{cmd}" >> {fish_config}',
    shell=True,
    check=True
)

print(f"✅ Token added to {fish_config}")
print("💡 Run `exec fish` or open a new terminal to load it.")
