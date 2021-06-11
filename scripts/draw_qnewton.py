from julia import QuadraticMap, QuadraticNewtonMap
import numpy as np
import cv2

qnewt = QuadraticNewtonMap(QuadraticMap(c=1j))
im = qnewt.draw_julia(res_x=600, res_y=600, x_range=(-2, 2), y_range=(-2, 2))
im = qnewt.draw_ray(im=im, line_weight=4,
                    x_range=(-2, 2), y_range=(-2, 2),
                    angles=[0, -2*np.pi/3, 2*np.pi/3])
im = qnewt.draw_eqpot(im=im, line_weight=2,
                      x_range=(-2, 2), y_range=(-2, 2))
cv2_im = np.array(im.convert('RGB'))[:,:,::-1]
cv2.imshow('hello', cv2_im)
cv2.waitKey(0)
im.show()
