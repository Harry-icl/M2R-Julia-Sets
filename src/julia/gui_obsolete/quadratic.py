"""Interactive GUI function for displaying Mandelbrot and Julia sets."""


def main_quadratic(multiprocessing: bool = False):
    """Run the interactive GUI."""
    import cv2
    import numpy as np
    from math import sqrt
    import PySimpleGUI as sg

    from .tools import to_complex, from_complex, title_generator_quad, \
        title_generator_quad_julia
    from .constants import X_RANGEM0, Y_RANGEM0, X_RANGEJ0, Y_RANGEJ0, \
        RESOLUTION, ITERATIONS, REC_COLOR, RAY_COLOR
    from julia import QuadraticMap

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
        x_res_m, y_res_m, x_res_j, y_res_j, external_rays_angles

    btn_down, drag = False, False
    x_range_m, y_range_m = X_RANGEM0, Y_RANGEM0
    x_range_j, y_range_j = X_RANGEJ0, Y_RANGEJ0

    start_coords = None
    x_res_m = RESOLUTION
    y_res_m = RESOLUTION
    x_res_j = RESOLUTION
    y_res_j = RESOLUTION
    external_rays_angles = []

    def draw_external_rays(angles):
        global open_cv_image_mandel, x_range_m, y_range_m, x_res_m, y_res_m
        for theta in angles:
            print(f"Drawing external ray at {theta}*2pi radians...")
            ray = [from_complex(z,
                                x_range_m, y_range_m,
                                x_res_m, y_res_m)
                   for z in quadratic_map.external_ray(theta)]
            pairs = zip(ray[:-1], ray[1:])

            for pair in pairs:
                cv2.line(open_cv_image_mandel,
                         pair[0], pair[1],
                         color=RAY_COLOR, thickness=1)
            cv2.imshow('mandel', open_cv_image_mandel)

    def click_event_mandel(event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        global btn_down, drag, x_range_m, y_range_m, start_coords, \
            open_cv_image_mandel, open_cv_image_julia, x_res_m, y_res_m, \
            external_rays_angles

        if event == cv2.EVENT_LBUTTONDOWN:
            btn_down = True
            start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not drag:
            btn_down = False

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
            pil_img_mandel = quadratic_map.draw_mandelbrot(res_x=x_res_m,
                                                           res_y=y_res_m,
                                                           iterations=ITERATIONS,  # noqa E501
                                                           x_range=x_range_m,
                                                           y_range=y_range_m,
                                                           multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
            cv2.imshow('mandel', open_cv_image_mandel)
            cv2.setWindowTitle('mandel',
                               title_generator_quad(x_range_m,
                                                    y_range_m))
            draw_external_rays(external_rays_angles)

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
            quadratic_map.c = to_complex(x, y,
                                         x_range_m, y_range_m,
                                         x_res_m, y_res_m)
            print(f"Recalculating julia set with {quadratic_map.c} as c...")
            pil_img_julia = quadratic_map.draw_julia(res_x=x_res_j,
                                                     res_y=y_res_j,
                                                     iterations=ITERATIONS,
                                                     x_range=x_range_j,
                                                     y_range=y_range_j,
                                                     multiprocessing=True)
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_quad_julia(quadratic_map.c,
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
            quadratic_map.c = to_complex(*start_coords,
                                         x_range_j, y_range_j,
                                         x_res_j, y_res_j)
            print(f"Recalculating with {quadratic_map.c} as c...")
            pil_img_julia = quadratic_map.draw_julia(res_x=x_res_j,
                                                     res_y=y_res_j,
                                                     iterations=ITERATIONS,
                                                     x_range=x_range_j,
                                                     y_range=y_range_j,
                                                     multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_quad_julia(quadratic_map.c,
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
            pil_img_julia = quadratic_map.draw_julia(res_x=x_res_j,
                                                     res_y=y_res_j,
                                                     iterations=ITERATIONS,  # noqa E501
                                                     x_range=x_range_j,
                                                     y_range=y_range_j,
                                                     multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_quad_julia(quadratic_map.c,
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

    quadratic_map = QuadraticMap(c=0)
    pil_img_mandel = quadratic_map.draw_mandelbrot(res_x=x_res_m,
                                                   res_y=y_res_m,
                                                   iterations=ITERATIONS,  # noqa E501
                                                   x_range=x_range_m,
                                                   y_range=y_range_m,
                                                   multiprocessing=multiprocessing)  # noqa E501
    open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
    quadratic_map.c = 0
    pil_img_julia = quadratic_map.draw_julia(res_x=x_res_j,
                                             res_y=y_res_j,
                                             iterations=ITERATIONS,
                                             x_range=x_range_j,
                                             y_range=y_range_j,
                                             multiprocessing=multiprocessing)
    open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
    cv2.imshow('mandel', open_cv_image_mandel)
    cv2.imshow('julia', open_cv_image_julia)
    cv2.setWindowTitle('mandel',
                       title_generator_quad(x_range_m, y_range_m))
    cv2.setWindowTitle('julia',
                       title_generator_quad_julia(quadratic_map.c,
                                                  x_range_j,
                                                  y_range_j))
    cv2.moveWindow('mandel', 0, 0)
    cv2.moveWindow('julia', RESOLUTION, 0)
    cv2.setMouseCallback('mandel', click_event_mandel)
    cv2.setMouseCallback('julia', click_event_julia)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('m'):
            print("Resetting Mandelbrot view...")
            x_res_m = RESOLUTION
            y_res_m = RESOLUTION
            x_range_m = X_RANGEM0
            y_range_m = Y_RANGEM0
            pil_img_mandel = quadratic_map.draw_mandelbrot(res_x=x_res_m,
                                                           res_y=y_res_m,
                                                           iterations=ITERATIONS,  # noqa E501
                                                           x_range=x_range_m,
                                                           y_range=y_range_m,
                                                           multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
            cv2.imshow('mandel', open_cv_image_mandel)
            cv2.setWindowTitle('mandel',
                               title_generator_quad(x_range_m,
                                                    y_range_m))
            draw_external_rays(external_rays_angles)

        elif key == ord('j'):
            print("Resetting Julia view...")
            x_res_j = RESOLUTION
            y_res_j = RESOLUTION
            x_range_j = X_RANGEJ0
            y_range_j = Y_RANGEJ0
            pil_img_julia = quadratic_map.draw_julia(res_x=x_res_m,
                                                     res_y=y_res_m,
                                                     iterations=ITERATIONS,
                                                     x_range=x_range_j,
                                                     y_range=y_range_j,
                                                     multiprocessing=multiprocessing)  # noqa E501
            open_cv_image_julia = np.array(pil_img_julia.convert('RGB'))
            cv2.imshow('julia', open_cv_image_julia)
            cv2.setWindowTitle('julia',
                               title_generator_quad_julia(quadratic_map.c,
                                                          x_range_j,
                                                          y_range_j))

        elif key == ord('e'):
            sg.theme('Material1')
            layout = [
                [sg.Text('Please enter the angle for the external ray as a mu'
                         'ltiple of 2pi (i.e. enter 1 to get 2pi radians).',
                         size=(50, 2))],
                [sg.Text('Theta', size=(10, 1)), sg.InputText(size=(10, 1)),
                 sg.Button('Draw Ray', size=(25, 1))],
                [sg.Text('Or enter the number of evenly-spaced rays you would '
                         'like to draw.', size=(50, 2))],
                [sg.Text('Rays', size=(10, 1)),
                 sg.InputText(size=(10, 1)),
                 sg.Button('Draw Rays', size=(25, 1))],
                [sg.Button('Remove all external rays', size=(22, 1)),
                 sg.Cancel(size=(23, 1))]
            ]
            window = sg.Window('External rays', layout)
            event, values = window.read()
            window.close()
            if event == sg.WIN_CLOSED or event == 'Cancel':
                continue
            elif event == 'Remove all external rays':
                print("Removing external rays...")
                external_rays_angles = []
                pil_img_mandel = quadratic_map.draw_mandelbrot(res_x=x_res_m,
                                                               res_y=y_res_m,
                                                               iterations=ITERATIONS,  # noqa E501
                                                               x_range=x_range_m,  # noqa E501
                                                               y_range=y_range_m,  # noqa E501
                                                               multiprocessing=multiprocessing)  # noqa E501
                open_cv_image_mandel = np.array(pil_img_mandel.convert('RGB'))
                cv2.imshow('mandel', open_cv_image_mandel)
                cv2.setWindowTitle('mandel',
                                   title_generator_quad(x_range_m,
                                                        y_range_m))
            if event == 'Draw Ray':
                try:
                    theta = float(values[0])
                except(ValueError):
                    print("Not a valid angle. Angles must be a float.")
                    continue
                external_rays_angles += [theta]
                draw_external_rays([theta])
            elif event == 'Draw Rays':
                try:
                    count = int(values[1])
                except(ValueError):
                    print("Not a valid number of rays. Number of rays must be "
                          "an integer.")
                    continue
                theta_list = list(np.linspace(0, 1, count))
                external_rays_angles += theta_list
                draw_external_rays(theta_list)
