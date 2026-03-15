"""Debug: check setuptools / pip / Python versions inside the active venv."""
import json, sys, time, importlib.metadata as im
from pathlib import Path

LOG = Path("/home/cbrissette/cuopt-opt/.cursor/debug-aca502.log")
LOG.parent.mkdir(parents=True, exist_ok=True)

def _v(pkg):
    try:
        return im.version(pkg)
    except im.PackageNotFoundError:
        return "NOT_FOUND"

def _has_backends():
    try:
        import setuptools.backends  # noqa: F401
        return True
    except ImportError:
        return False

entries = [
    {"sessionId":"aca502","hypothesisId":"H-A","location":"debug_env_check.py:setuptools",
     "message":"setuptools version","data":{"setuptools":_v("setuptools")},"timestamp":int(time.time()*1000)},
    {"sessionId":"aca502","hypothesisId":"H-A","location":"debug_env_check.py:has_backends",
     "message":"setuptools.backends importable","data":{"importable":_has_backends()},"timestamp":int(time.time()*1000)},
    {"sessionId":"aca502","hypothesisId":"H-C","location":"debug_env_check.py:pip",
     "message":"pip version","data":{"pip":_v("pip")},"timestamp":int(time.time()*1000)},
    {"sessionId":"aca502","hypothesisId":"H-B","location":"debug_env_check.py:python",
     "message":"python version","data":{"python":sys.version,"prefix":sys.prefix},"timestamp":int(time.time()*1000)},
]

with LOG.open("a") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print("Diagnostic written to", LOG)
for e in entries:
    print(f"  [{e['hypothesisId']}] {e['message']}: {e['data']}")
