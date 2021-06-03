if __name__ == "__main__":
    import time
    time0 = time.time()
    from julia import QuadraticMap, QuadraticNewtonMap

    mandelbrot = QuadraticMap()
    newton = QuadraticNewtonMap(mandelbrot)
    im = newton.draw_mandelbrot(res_x=1080, res_y=1080, x_range=(-3, 3), y_range=(-3, 3), multiprocessing=True)
    im.show()
    print(time.time() - time0)