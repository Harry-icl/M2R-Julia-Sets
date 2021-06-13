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
    parser.add_argument('-n', '--newton',
                        help="Use the newton mapping of the function.",
                        action='store_true')
    parser.add_argument('-m', '--multiprocessing',
                        help="Use parallelisation - off by default.",
                        action='store_true')
    parser.add_argument('-p', '--preimages',
                        help="Draw 10 preimages instead of julia sets",
                        action='store_true')
    args = parser.parse_args()
    if args.newton and args.cubic:
        from .gui_rebuild import CubicNewtonWindows
        cubic_newton_gui = CubicNewtonWindows(
            multiprocessing=args.multiprocessing)
        cubic_newton_gui.start()
    elif args.newton:
        from .gui_rebuild import QuadraticNewtonWindows
        quadratic_newton_gui = QuadraticNewtonWindows(
            multiprocessing=args.multiprocessing)
        quadratic_newton_gui.start()
    elif args.cubic:
        from .gui_rebuild import CubicWindows
        cubic_gui = CubicWindows(multiprocessing=args.multiprocessing,
                                 preimages=args.preimages)
        cubic_gui.start()
    else:
        from .gui_rebuild import QuadraticWindows
        quadratic_gui = QuadraticWindows(multiprocessing=args.multiprocessing,
                                         preimages=args.preimages)
        quadratic_gui.start()
