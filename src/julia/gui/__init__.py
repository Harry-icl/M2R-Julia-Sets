import sys
if sys.platform == "darwin":
    from .main_macos import main
elif sys.platform == "win32":
    from .main_win import main
else:
    raise SystemError("Platform not supported.")
