if __name__ == "__main__":
    RESOLUTION = 480
    ITERATIONS = 50
    import cv2
    import numpy as np
    from julia import CubicMap
    from math import sqrt

    btn_down, drag = False, False
    x_range, y_range = (-2, 2), (-2, 2)
    start_coords = None
    x_res = RESOLUTION
    y_res = RESOLUTION

    def to_complex(x, y):
        global x_range, y_range
        x_val = x_range[0] + ((x / x_res) * (x_range[1] - x_range[0]))
        y_val = y_range[0] + (((y_res - y) / y_res) * (y_range[1] - y_range[0]))
        return complex(x_val, y_val)
    
    def title_generator(a, x_range, y_range):
        function_name = (f"z^3 - ({round(a.real, 3)} + {round(a.imag, 3)}i)z + b" if a.imag > 0
                         else f"z^3 - ({round(a.real, 3)} {round(a.imag, 3)}i)z + b")
        bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i" if y_range[0] > 0
                       else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
        top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i" if y_range[1] > 0
                       else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
        return f"{function_name}, ({bottom_left}, {top_right})"

    def click_event(event, x, y, flags, params):
        global btn_down, drag, x_range, y_range, start_coords, open_cv_image, x_res, y_res

        if event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True
            start_coords = (x, y)
            cv2.waitKey(2)

        if event == cv2.EVENT_LBUTTONUP and not drag:
            btn_down = False
            cubic_map.a = to_complex(*start_coords)
            print(f"Recalculating with {cubic_map.a} as a...")
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res, res_y=y_res, iterations=ITERATIONS, x_range=x_range, y_range=y_range, multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets', title_generator(cubic_map.a, x_range, y_range))
        
        if event == cv2.EVENT_LBUTTONUP and drag:
            btn_down = False
            drag = False
            start_num = to_complex(*start_coords)
            end_num = to_complex(x, y)
            x_range = (min(start_num.real, end_num.real), max(start_num.real, end_num.real))
            y_range = (min(start_num.imag, end_num.imag), max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {x_range}, y: {y_range}...")
            ratio = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0]) if y_range[0] != y_range[1] else 1
            x_res = int(RESOLUTION*sqrt(ratio))
            y_res = int(RESOLUTION/sqrt(ratio))
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res, res_y=y_res, iterations=ITERATIONS, x_range=x_range, y_range=y_range, multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets', title_generator(cubic_map.a, x_range, y_range))


        if event == cv2.EVENT_MOUSEMOVE and btn_down:
            drag = True
            rectangle_open_cv_image = open_cv_image.copy()
            cv2.rectangle(rectangle_open_cv_image, start_coords, (x, y), (255, 0, 0), 2)
            cv2.imshow('M2R-Julia-Sets', rectangle_open_cv_image)

    
    cubic_map = CubicMap(a=0)
    pil_img = cubic_map.draw_mandelbrot(res_x=x_res, res_y=y_res, iterations=ITERATIONS, x_range=x_range, y_range=y_range, multiprocessing=True)
    open_cv_image = np.array(pil_img.convert('RGB'))
    cv2.imshow('M2R-Julia-Sets', open_cv_image)
    cv2.setWindowTitle('M2R-Julia-Sets', title_generator(cubic_map.a, x_range, y_range))
    cv2.setMouseCallback('M2R-Julia-Sets', click_event)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('r'):
            x_res = RESOLUTION
            y_res = RESOLUTION
            x_range = (-2, 2)
            y_range = (-2, 2)
            pil_img = cubic_map.draw_mandelbrot(res_x=x_res, res_y=y_res, iterations=ITERATIONS, x_range=x_range, y_range=y_range, multiprocessing=True)
            open_cv_image = np.array(pil_img.convert('RGB'))
            cv2.imshow('M2R-Julia-Sets', open_cv_image)
            cv2.setWindowTitle('M2R-Julia-Sets', title_generator(cubic_map.a, x_range, y_range))
