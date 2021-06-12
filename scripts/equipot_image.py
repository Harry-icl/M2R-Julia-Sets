from julia import QuadraticMap
from PIL import Image
import numpy as np

qm = QuadraticMap(c=-1)

equipot_lines_drawings = [
    qm.draw_equipotential(0.05, res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)),
    qm.draw_equipotential(0.1, res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2)),
    qm.draw_equipotential(0.2, res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2))
]

equipot_arrays = [np.array(drawing.convert('RGB')) for drawing in equipot_lines_drawings]

jul = qm.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2))

jul_array = np.array(jul.convert('RGB'))

for equipot_array in equipot_arrays:
    jul_array = np.minimum(equipot_array, jul_array)

final_im = Image.fromarray(jul_array)
final_im.show()