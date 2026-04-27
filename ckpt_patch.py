"""
Patch for cross-platform .ckpt loading.
Checkpoints saved on Linux use PosixPath internally.
On Windows, torch.load fails to unpickle them.
This patch remaps PosixPath -> WindowsPath before loading.
"""
import pathlib
import sys

def apply_posixpath_patch():
    """
    On Windows: remap pathlib.PosixPath to pathlib.WindowsPath so that
    checkpoints pickled on Linux can be loaded with torch.load.
    Safe to call multiple times (idempotent).
    """
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
