if __name__ == "__main__":
    import cProfile
    from pstats import SortKey, Stats
    import io
    import time
    #pr = cProfile.Profile(builtins=False)

    from julia import CubicMap
    
    time0 = time.time()
    threebrot = CubicMap(a=complex(1, 1))
    #pr.enable()
    img = threebrot.draw_mandelbrot(res_x=2048, res_y=2048, iterations=100, x_range=(-2, 2), y_range=(-2, 2), z_max=2, multiprocessing=True)
    img.show()
    #pr.disable()
    threebrot.b = complex(1/2)
    threebrot.draw_julia(res_x=2048, res_y=2048, multiprocessing=True)
    
    print(time.time() - time0)
    #s = io.StringIO()
    #ps = Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    #ps.print_stats(30)
    #print(s.getvalue())