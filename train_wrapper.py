#!/usr/bin/env python3
"""
Wrapper script to set environment variables before importing any modules
to prevent the '_Deterministic' pickle error with submitit.
"""

import os

# Set environment variables before any torch imports
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TORCH_DETERMINISTIC"] = "0"
os.environ["PYTORCH_DISABLE_DETERMINISTIC_ALGORITHMS"] = "1"

# Now import and run the main training script
from boa.train import main

if __name__ == "__main__":
    main()
