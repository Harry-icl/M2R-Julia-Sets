from julia import CubicMap

threebrot = CubicMap(a=0)
threebrot.draw_multibrot(res_x=2048, res_y=2048, iterations=100, x_range=(-2, 2), y_range=(-2, 2), z_max=2)
threebrot.b = complex(1/2)
threebrot.draw_julia()