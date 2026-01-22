import sys
import os

# Get path to the root 'simcode' directory (parent of 'tests')
# __file__ is inside tests/, so we go up one level
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root path to sys.path
if root_path not in sys.path:
    sys.path.insert(0, root_path)