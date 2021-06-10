"""Interactive GUI function for displaying Mandelbrot and Julia sets."""


def main_cubic(multiprocessing: bool = False):
    """Run the interactive GUI."""
    import cv2
    import numpy as np
    from math import sqrt
    import PySimpleGUI as sg

    from .tools import to_complex, from_complex, title_generator, \
        title_generator_julia
    from .constants import X_RANGEM0, Y_RANGEM0, X_RANGEJ0, Y_RANGEJ0, \
        RESOLUTION, ITERATIONS, REC_COLOR, RAY_COLOR
    from julia import CubicMap

    root = sg.tk.Tk()
    root.withdraw()
    # DO NOT REMOVE THESE LINES
    # These lines don't do anything, but if you remove them then it breaks the
    # program on macos

    cv2.namedWindow('Loading...')
    cv2.setWindowProperty("Loading...",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    cv2.setWindowProperty("Loading...",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_NORMAL)
    cv2.destroyWindow("Loading...")
    # KEEP THESE TOO
    # These lines also don't do anything, but they make sure the window appears
    # in focus.

    sg.SetOptions(font='Helvetica 15', border_width=5)

    global btn_down, drag, x_range_m, y_range_m, x_range_j, y_range_j, \
        start_coords, open_cv_image_mandel, open_cv_image_julia, \
        x_res_m, y_res_m, x_res_j, y_res_j

    btn_down, drag = False, False
    x_range_m, y_range_m = X_RANGEM0, Y_RANGEM0
    x_range_j, y_range_j = X_RANGEJ0, Y_RANGEJ0

    start_coords = None
    x_res_m = RESOLUTION
    y_res_m = RESOLUTION
    x_res_j = RESOLUTION
    y_res_j = RESOLUTION
    external_rays = False

    def click_event_mandel(event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        global btn_down, drag, x_range_m, y_range_m, start_coords, \
            open_cv_image_mandel, open_cv_image_julia, x_res_m, y_res_m

        if event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True
            start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not drag:
            btn_down = False
            cubic_map.a = to_complex(*start_coords,
                                     x_range_m, y_range_m,
                                     x_res_m, y_res_m)
            print(f"Recalculating with {cubic_map.a} as a...")
            pil_img_mandel = cubic_map.draw_mandelbrot(res_x=x_res_m,
                                                       res_y=y_res_m,
                                                       iterations=ITERATIONS,
                                                       x_range=x_range_m,
                                                       y_range=y_range_m,
                                                       multiprocessing=multiprocessing)  # noqa E501
            pil_img_julia = cubic_map.draw_julia(res_x=x_res_j,
                                                 res_y=y_res_j,
                                                 iterations=ITERATIONS,
                                                 x_range=x_range_j,
                                                 y_range=y_range_j,
                                                 multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('mandel', open_cv_image_mandel)
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('mandel',
                               title_generator(cubic_map.a,
                                               x_range_m,
                                               y_range_m))
            cv2.setWindowTitle('julia',
                               title_generator_julia(cubic_map.a,
                                                     cubic_map.b,
                                                     x_range_m,
                                                     y_range_m))

        elif event == cv2.EVENT_LBUTTONUP and drag:
            btn_down = False
            drag = False
            start_num = to_complex(*start_coords,
                                   x_range_m, y_range_m,
                                   x_res_m, y_res_m)
            end_num = to_complex(x, y, x_range_m, y_range_m, x_res_m, y_res_m)
            x_range_m = (min(start_num.real, end_num.real),
                         max(start_num.real, end_num.real))
            y_range_m = (min(start_num.imag, end_num.imag),
                         max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {x_range_m}, y: {y_range_m}...")
            ratio = ((x_range_m[1] - x_range_m[0])
                     / (y_range_m[1] - y_range_m[0])
                     if y_range_m[0] != y_range_m[1]
                     else 1)
            x_res_m = int(RESOLUTION*sqrt(ratio))
            y_res_m = int(RESOLUTION/sqrt(ratio))
            pil_img_mandel = cubic_map.draw_mandelbrot(res_x=x_res_m,
                                                       res_y=y_res_m,
                                                       iterations=ITERATIONS,
                                                       x_range=x_range_m,
                                                       y_range=y_range_m,
                                                       multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
            cv2.imshow('mandel', open_cv_image_mandel)
            cv2.setWindowTitle('mandel',
                               title_generator(cubic_map.a,
                                               x_range_m,
                                               y_range_m))

        elif event == cv2.EVENT_MOUSEMOVE and btn_down:
            drag = True
            rectangle_open_cv_image_mandel = open_cv_image_mandel.copy()
            cv2.rectangle(rectangle_open_cv_image_mandel,
                          pt1=start_coords,
                          pt2=(x, y),
                          color=REC_COLOR,
                          thickness=2)
            cv2.imshow('mandel', rectangle_open_cv_image_mandel)

        elif event == cv2.EVENT_RBUTTONDOWN:
            cubic_map.b = to_complex(x, y,
                                     x_range_m, y_range_m,
                                     x_res_m, y_res_m)
            print(f"Recalculating julia set with {cubic_map.b} as b...")
            pil_img_julia = cubic_map.draw_julia(res_x=x_res_j,
                                                 res_y=y_res_j,
                                                 iterations=ITERATIONS,
                                                 x_range=x_range_j,
                                                 y_range=y_range_j,
                                                 multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_julia(cubic_map.a,
                                                     cubic_map.b,
                                                     x_range_j,
                                                     y_range_j))

    def click_event_julia(event, x, y, flags, params):
        """Process mouse interaction in julia set window."""
        global btn_down, drag, x_range_j, y_range_j, start_coords, \
            open_cv_image_julia, x_res_j, y_res_j

        if event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True
            start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not drag:
            btn_down = False
            cubic_map.a = to_complex(*start_coords,
                                     x_range_j, y_range_j,
                                     x_res_j, y_res_j)
            print(f"Recalculating with {cubic_map.a} as a...")
            pil_img_julia = cubic_map.draw_julia(res_x=x_res_j,
                                                 res_y=y_res_j,
                                                 iterations=ITERATIONS,
                                                 x_range=x_range_j,
                                                 y_range=y_range_j,
                                                 multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_julia(cubic_map.a,
                                                     x_range_j,
                                                     y_range_j))

        elif event == cv2.EVENT_LBUTTONUP and drag:
            btn_down = False
            drag = False
            start_num = to_complex(*start_coords,
                                   x_range_j, y_range_j,
                                   x_res_j, y_res_j)
            end_num = to_complex(x, y, x_range_j, y_range_j, x_res_j, y_res_j)
            x_range_j = (min(start_num.real, end_num.real),
                         max(start_num.real, end_num.real))
            y_range_j = (min(start_num.imag, end_num.imag),
                         max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {x_range_j}, y: {y_range_j}...")
            ratio = ((x_range_j[1] - x_range_j[0])
                     / (y_range_j[1] - y_range_j[0])
                     if y_range_j[0] != y_range_j[1]
                     else 1)
            x_res_j = int(RESOLUTION*sqrt(ratio))
            y_res_j = int(RESOLUTION/sqrt(ratio))
            pil_img_julia = cubic_map.draw_julia(res_x=x_res_j,
                                                 res_y=y_res_j,
                                                 iterations=ITERATIONS,
                                                 x_range=x_range_j,
                                                 y_range=y_range_j,
                                                 multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_julia(cubic_map.a,
                                                     cubic_map.b,
                                                     x_range_j,
                                                     y_range_j))

        elif event == cv2.EVENT_MOUSEMOVE and btn_down:
            drag = True
            rectangle_open_cv_image_julia = open_cv_image_julia.copy()
            cv2.rectangle(rectangle_open_cv_image_julia,
                          pt1=start_coords,
                          pt2=(x, y),
                          color=REC_COLOR,
                          thickness=2)
            cv2.imshow('julia', rectangle_open_cv_image_julia)

    cubic_map = CubicMap(a=0)
    pil_img_mandel = cubic_map.draw_mandelbrot(res_x=x_res_m,
                                               res_y=y_res_m,
                                               iterations=ITERATIONS,
                                               x_range=x_range_m,
                                               y_range=y_range_m,
                                               multiprocessing=multiprocessing)
    open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
    cubic_map.b = 0
    pil_img_julia = cubic_map.draw_julia(res_x=x_res_j,
                                         res_y=y_res_j,
                                         iterations=ITERATIONS,
                                         x_range=x_range_j,
                                         y_range=y_range_j,
                                         multiprocessing=multiprocessing)
    open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
    cv2.imshow('mandel', open_cv_image_mandel)
    cv2.imshow('julia', open_cv_image_julia)
    cv2.setWindowTitle('mandel',
                       title_generator(cubic_map.a, x_range_m, y_range_m))
    cv2.setWindowTitle('julia',
                       title_generator_julia(cubic_map.a,
                                             cubic_map.b,
                                             x_range_j,
                                             y_range_j))
    cv2.moveWindow('mandel', 0, 0)
    cv2.moveWindow('julia', RESOLUTION, 0)
    cv2.setMouseCallback('mandel', click_event_mandel)
    cv2.setMouseCallback('julia', click_event_julia)
    cv2.setWindowProperty('mandel', cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty('julia', cv2.WND_PROP_TOPMOST, 1)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('m'):
            x_res_m = RESOLUTION
            y_res_m = RESOLUTION
            x_range_m = X_RANGEM0
            y_range_m = Y_RANGEM0
            pil_img_mandel = cubic_map.draw_mandelbrot(res_x=x_res_m,
                                                       res_y=y_res_m,
                                                       iterations=ITERATIONS,
                                                       x_range=x_range_m,
                                                       y_range=y_range_m,
                                                       multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
            cv2.imshow('mandel', open_cv_image_mandel)
            cv2.setWindowTitle('mandel',
                               title_generator(cubic_map.a,
                                               x_range_m,
                                               y_range_m))

        elif key == ord('j'):
            x_res_j = RESOLUTION
            y_res_j = RESOLUTION
            x_range_j = X_RANGEJ0
            y_range_j = Y_RANGEJ0
            pil_img_julia = cubic_map.draw_julia(res_x=x_res_m,
                                                 res_y=y_res_m,
                                                 iterations=ITERATIONS,
                                                 x_range=x_range_j,
                                                 y_range=y_range_j,
                                                 multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_julia(cubic_map.a,
                                                     cubic_map.b,
                                                     x_range_j,
                                                     y_range_j))

        elif key == ord('e'):
            if not external_rays:
                external_ray_open_cv_image = open_cv_image_julia.copy()
                zero_ray = [from_complex(z)
                            for z in cubic_map.external_ray(0)]
                pi_3_ray = [from_complex(z)
                            for z in cubic_map.external_ray(1/6)]
                pi_ray = [from_complex(z)
                          for z in cubic_map.external_ray(1/2)]
                pairs = zip(zero_ray[:-1] + pi_3_ray[:-1] + pi_ray[:-1],
                            zero_ray[1:] + pi_3_ray[1:] + pi_ray[1:])

                for pair in pairs:
                    cv2.line(external_ray_open_cv_image,
                             pair[0], pair[1],
                             color=RAY_COLOR, thickness=1)

                cv2.imshow('julia', external_ray_open_cv_image)
