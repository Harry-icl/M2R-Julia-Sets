if __name__ == "__main__":
    import time
    time0 = time.time()
    from julia import QuadraticMap

    mandelbrot = QuadraticMap()
    im = mandelbrot.draw_mandelbrot(res_x=2048, res_y=2048, x_range=(-2, 2), y_range=(-2, 2), multiprocessing=True)
    im.show()
    print(time.time() - time0)