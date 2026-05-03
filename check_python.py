import sys

def _enforce():
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print("\n❌ ERROR: Python 3.10 is required for this project.")
        print(f"[!] Current version: {sys.version.split()[0]}")
        print("[!] Please install Python 3.10 and recreate your virtual environment.\n")
        sys.exit(1)

# Run automatically when imported
_enforce()
