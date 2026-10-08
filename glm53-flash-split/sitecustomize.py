# tf-split loader: installs the split only inside `tensorfold serve`; every other Python process is untouched.
# Put this directory on the server's PYTHONPATH (e.g. in TensorFold's serve env file). TF_SPLIT=0 disables it.
import sys
def _tf_split_boot():
    argv = list(getattr(sys, "argv", []) or [])
    if not argv or "tensorfold" not in argv[0] or "serve" not in argv:
        return
    import os
    if os.environ.get("TF_SPLIT", "1") == "0":
        return
    try:
        import tf_split
        tf_split.install()
    except Exception as exc:  # never stop the server for this
        sys.stderr.write(f"[tf-split] install failed: {exc!r}\n")
_tf_split_boot()
