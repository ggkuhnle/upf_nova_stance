"""Run the pipeline: abstracts -> stance -> network. Usage: python run_all.py [--backend llm]"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent / "pipeline"
extra = sys.argv[1:]
for step, args in [("fetch_abstracts.py", []), ("classify_stance.py", extra), ("build_network.py", [])]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(here / step), *args], check=True)
