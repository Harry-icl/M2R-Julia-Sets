from julia import CubicMap

cubic_map = CubicMap(a=complex(-1))
im = cubic_map.draw_mandelbrot(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2))
im.show()
cubic_map.b = complex(0, 1)
im2 = cubic_map.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2))
im2.show()