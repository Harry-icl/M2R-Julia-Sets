if __name__ == "__main__":
    import sys
    if sys.platform == "darwin":
        from .main_macos import main
        main()
    elif sys.platform == "win32":
        from .main_win import main
        main()
    else:
        raise SystemError("Platform not supported.")
