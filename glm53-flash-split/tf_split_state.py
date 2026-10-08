"""Shared state for tf_split (not reloaded)."""
import threading
STATS = {"splits": 0, "failures": 0, "last": None, "last_error": None}
LOCK = threading.Lock()
EXTRA_STARTS = {}
STAMP = [None]
PENDING = {}
BUDGET = [0]
BUDGET_COND = threading.Condition()
