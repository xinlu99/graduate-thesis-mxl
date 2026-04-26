"""
Master script: generates all data CSVs and figures for Chapter 3.
Run: python generate_figures_and_tables.py
"""
import os, subprocess, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== Step 1: Generating data ===")
subprocess.check_call([sys.executable, os.path.join(SCRIPT_DIR, 'generate_data.py')])

print("\n=== Step 2: Generating figures ===")
subprocess.check_call([sys.executable, os.path.join(SCRIPT_DIR, 'generate_figures.py')])

print("\n=== All done ===")
