from julia import QuadraticMap, QuadraticNewtonMap

quad_newt = QuadraticNewtonMap(QuadraticMap(c=1j))
quad_newt.draw_rays(res_x=2048, res_y=2048, line_weight=10, multiples=11)
quad_newt.draw_eqpots(res_x=2048, res_y=2048, line_weight=10)
