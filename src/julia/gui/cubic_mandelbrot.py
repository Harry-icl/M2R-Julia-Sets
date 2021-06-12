"""Interactive GUI function for displaying Mandelbrot sets."""


def main():
    """Run the interactive GUI."""
    import cv2
    import numpy as np
    from math import sqrt

    from .tools import to_complex, title_generator
    from .constants import X_RANGE0, Y_RANGE0, RESOLUTION, ITERATIONS, \
        REC_COLOR
    from julia import CubicMap

    global btn_down, drag, x_range, y_range, start_coords, open_cv_image, \
        x_res, y_res

    btn_down, drag = False, False
    x_range, y_range = X_RANGE0, Y_RANGE0
    start_coords = None
    x_res = RESOLUTION
    y_res = RESOLUTION

    def click_event(event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        global btn_down, drag, x_range, y_range, start_coords, open_cv_image, \
            x_res, y_res

        if event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True
            start_coords = (x, y)
            cv2.waitKey(5)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        if event == cv2.EVENT_LBUTTONUP and not drag:
            btn_down = False
            cubic_map.a = to_complex(*start_coords,
                                     x_range, y_range,
                                     x_res, y_res)
            print(f"Recalculating with {cubic_map.a} as a...")
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res,
                                                res_y=y_res,
                                                iterations=ITERATIONS,
                                                x_range=x_range,
                                                y_range=y_range,
                                                multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets',
                               title_generator(cubic_map.a, x_range, y_range))

        if event == cv2.EVENT_LBUTTONUP and drag:
            btn_down = False
            drag = False
            start_num = to_complex(*start_coords,
                                   x_range, y_range,
                                   x_res, y_res)
            end_num = to_complex(x, y, x_range, y_range, x_res, y_res)
            x_range = (min(start_num.real, end_num.real),
                       max(start_num.real, end_num.real))
            y_range = (min(start_num.imag, end_num.imag),
                       max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {x_range}, y: {y_range}...")
            ratio = ((x_range[1] - x_range[0])
                     / (y_range[1] - y_range[0])
                     if y_range[0] != y_range[1]
                     else 1)
            x_res = int(RESOLUTION*sqrt(ratio))
            y_res = int(RESOLUTION/sqrt(ratio))
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res,
                                                res_y=y_res,
                                                iterations=ITERATIONS,
                                                x_range=x_range,
                                                y_range=y_range,
                                                multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets',
                               title_generator(cubic_map.a, x_range, y_range))

        if event == cv2.EVENT_MOUSEMOVE and btn_down:
            drag = True
            rectangle_open_cv_image = open_cv_image.copy()
            cv2.rectangle(rectangle_open_cv_image,
                          pt1=start_coords,
                          pt2=(x, y),
                          color=REC_COLOR,
                          thickness=2)
            cv2.imshow('M2R-Julia-Sets', rectangle_open_cv_image)

    cubic_map = CubicMap(a=0)
    pil_img = cubic_map.draw_mandelbrot(res_x=x_res,
                                        res_y=y_res,
                                        iterations=ITERATIONS,
                                        x_range=x_range,
                                        y_range=y_range,
                                        multiprocessing=True)
    open_cv_image = np.array(pil_img.convert('RGB'))
    cv2.imshow('M2R-Julia-Sets', open_cv_image)
    cv2.setWindowTitle('M2R-Julia-Sets',
                       title_generator(cubic_map.a, x_range, y_range))
    cv2.setMouseCallback('M2R-Julia-Sets', click_event)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('r'):
            x_res = RESOLUTION
            y_res = RESOLUTION
            x_range = X_RANGE0
            y_range = Y_RANGE0
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res,
                                                res_y=y_res,
                                                iterations=ITERATIONS,
                                                x_range=x_range,
                                                y_range=y_range,
                                                multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets',
                               title_generator(cubic_map.a,
                                               x_range,
                                               y_range))
