"""Legacy module alias for unchanged training callers; use src.config.settings."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.config.settings")
