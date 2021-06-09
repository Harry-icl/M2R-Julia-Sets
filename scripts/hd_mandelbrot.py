if __name__ == "__main__":
    from julia import CubicMap
    from matplotlib import cm

    multibrot = CubicMap(a=0)
    img = multibrot.draw_mandelbrot(res_x=4096, res_y=4096, iterations=100, x_range=(-2, 2), y_range=(-2, 2), multiprocessing=True, colormap=cm.jet_r)
    img.show()