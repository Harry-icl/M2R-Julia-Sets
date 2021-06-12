from julia import CubicMap

cm = CubicMap(a=0, b=complex(0.456, 0.627))

cm.draw_julia(res_x=1440, res_y=1440, iterations=100, x_range=(-2, 2), y_range=(-2, 2)).show()

cm.b = complex(0.427, 1.187)

cm.draw_julia(res_x=1440, res_y=1440, iterations=50, x_range=(-2, 2), y_range=(-2, 2)).show()