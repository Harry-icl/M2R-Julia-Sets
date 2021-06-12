from julia import CubicMap

cubic_map = CubicMap(a=0, b=1j)

cubic_map.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)).show()

cubic_map.b = 0.16 + 0.847j

cubic_map.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)).show()

cubic_map.b = 0.24 + 0.827j

cubic_map.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)).show()

cubic_map.b = 0.562 - 0.417j

cubic_map.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)).show()