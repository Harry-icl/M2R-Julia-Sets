if __name__ == "__main__":
    import time
    
    from julia import QuadraticMap

    mandelbrot = QuadraticMap()
    im = mandelbrot.draw_mandelbrot(res_x=400, res_y=400, iterations=50, x_range=(-2, 2), y_range=(-2, 2), multiprocessing=True)
    time0 = time.time()
    for _ in range(10):
        im = mandelbrot.draw_mandelbrot(res_x=400, res_y=400, iterations=50, x_range=(-2, 2), y_range=(-2, 2), multiprocessing=True)
        mandelbrot.c = complex(-1/2, 1/2)
        jul = mandelbrot.draw_julia(multiprocessing=True)
    jul.show()
    print(time.time() - time0)