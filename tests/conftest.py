import os
import sys

# Run the tests from the repository root, where app.py expects artifacts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
