"""Interactive GUI."""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=("Draw Mandelbrot and Julia "
                                                  "sets."))
    parser.add_argument('-c', '--cubic',
                        help="Use the cubic function z^3 - az + b.",
                        action='store_true')
    parser.add_argument('-q', '--quadratic',
                        help="Use the quadratic function z^2 + c.",
                        action='store_true')
    parser.add_argument('-m', '--multiprocessing',
                        help="Use parallelisation - off by default.",
                        action='store_true')
    args = parser.parse_args()
    if args.cubic:
        from .gui import main_cubic
        main_cubic(args.multiprocessing)
    else:
        from .gui import main_quadratic
        main_quadratic(args.multiprocessing)
