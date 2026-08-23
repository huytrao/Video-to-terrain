#!/usr/bin/env python3
"""
File Sync & Workflow Example for Kaggle Runtime Toolkit
Demonstrates remote script execution, file generation, and download.
"""
import os
import sys
from pathlib import Path

# Add scripts dir to sys.path
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))
from kaggle_executor import KaggleExecutor, resolve_proxy_url

def main():
    proxy = resolve_proxy_url()
    if not proxy:
        print("❌ Error: Proxy URL could not be resolved from .env or environment.")
        sys.exit(1)
        
    print(f"Connecting to Kaggle proxy: {proxy[:55]}...")
    ex = KaggleExecutor(proxy)
    kid = ex.get_or_create_kernel()
    print(f"Active kernel ID: {kid}")
    
    # 1. Execute remote code that creates a sample file
    code = """
import json
data = {
    "status": "success",
    "device": "remote_kaggle_gpu",
    "items_processed": 100,
    "timestamp": 1724053200
}
with open('/kaggle/working/output_sample.json', 'w') as f:
    json.dump(data, f, indent=2)
print("✓ Created /kaggle/working/output_sample.json")
"""
    print("\n--- Executing Remote Task ---")
    ex.execute(kid, code)
    
    # 2. Download generated output
    local_dest = Path("./output_sample.json")
    print(f"\n--- Downloading Artifact ---")
    if ex.download_file("output_sample.json", str(local_dest)):
        print(f"✓ Downloaded successfully to: {local_dest.resolve()}")
        if local_dest.is_file():
            print("Content:")
            print(local_dest.read_text())
            local_dest.unlink(missing_ok=True)
    else:
        print("❌ Download failed.")

if __name__ == "__main__":
    main()
