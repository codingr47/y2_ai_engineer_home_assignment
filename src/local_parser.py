"""Legacy module alias for unchanged training callers; use src.frozen_fusion.local_parser."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.frozen_fusion.local_parser")
