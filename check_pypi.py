import urllib.request
import json
import re

def check_package(line):
    # Ignore comments and empty lines
    line = line.split('#')[0].strip()
    if not line:
        return
    
    # Extract package name and version
    match = re.match(r'^([a-zA-Z0-9_\-]+)(?:([=<>!~]+)(.*))?$', line)
    if not match:
        print(f"Skipping unparseable line: {line}")
        return
        
    pkg_name = match.group(1)
    version_op = match.group(2)
    version = match.group(3)
    
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        if version and version_op == '==':
            if version in data['releases']:
                print(f"✅ {pkg_name}=={version} exists")
            else:
                print(f"❌ {pkg_name}=={version} NOT FOUND (available: {list(data['releases'].keys())[:5]}...)")
        else:
            print(f"✅ {pkg_name} exists")
            
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ PACKAGE NOT FOUND: {pkg_name}")
        else:
            print(f"⚠️ Error checking {pkg_name}: {e}")
    except Exception as e:
        print(f"⚠️ Error checking {pkg_name}: {e}")

print("--- Checking scripts/requirements_trainer.txt ---")
with open('scripts/requirements_trainer.txt') as f:
    for line in f:
        check_package(line)

print("\n--- Checking scripts/requirements_pi.txt ---")
with open('scripts/requirements_pi.txt') as f:
    for line in f:
        check_package(line)
