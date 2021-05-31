from julia import CubicMap

multibrot = CubicMap(a=complex(1, 1))
multibrot.draw_mandelbrot(res_x=2048, res_y=2048, iterations=300, x_range=(-2, 2), y_range=(-2, 2))