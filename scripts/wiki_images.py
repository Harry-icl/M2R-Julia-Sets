from julia import QuadraticMap

quadratic_map = QuadraticMap(c=0)
man = quadratic_map.draw_mandelbrot(res_x=1440, res_y=1440)
jul = quadratic_map.draw_julia(res_x=1440, res_y=1440)

man.show()
jul.show()