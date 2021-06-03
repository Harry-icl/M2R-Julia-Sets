if __name__ == "__main__":
    from julia import CubicMap

    multibrot = CubicMap(a=0)
    img = multibrot.draw_mandelbrot(res_x=4096, res_y=4096, iterations=300, x_range=(-2, 2), y_range=(-2, 2), multiprocessing=True)
    img.show()