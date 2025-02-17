import os
import sys

# Add leaspy source to path (overwrite any existing leaspy package by inserting instead of appending)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
