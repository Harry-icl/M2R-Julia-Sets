"""Module containing the QuadraticWindows class."""
import numpy as np
from math import sqrt, pi

from julia.quadratic_map import QuadraticMap

from .constants import (X_RANGEM0, Y_RANGEM0, X_RANGEJ0, Y_RANGEJ0, RESOLUTION,
                        ITERATIONS, REC_COLOR)


class QuadraticWindows:
    """The class for the Quadratic GUI."""

    def __init__(self, multiprocessing: bool = False, preimages: bool = False):
        self.multiprocessing = multiprocessing
        if preimages:
            global ITERATIONS
            ITERATIONS = 10

        self.btn_down, self.drag = False, False
        self.x_range_m, self.y_range_m = X_RANGEM0, Y_RANGEM0
        self.x_range_j, self.y_range_j = X_RANGEM0, Y_RANGEM0

        self.start_coords = None
        self.x_res_m, self.y_res_m = RESOLUTION, RESOLUTION
        self.x_res_j, self.y_res_j = RESOLUTION, RESOLUTION

        self.external_rays_angles = []
        self.external_rays_angles_julia = []
        self.equipotentials = []

        self.quadratic_map = QuadraticMap(c=0)

    def start(self):
        """Start the quadratic GUI."""
        self._refresh_mandel()
        self._refresh_julia()

    def _title_generator(self):
        func_name = "z^2 + c"
        bottom_left = ((f"{round(self.x_range_m[0], 3)} + "
                        f"{round(self.y_range_m[0], 3)}i")
                       if self.y_range_m[0] >= 0
                       else (f"{round(self.x_range_m[0], 3)} "
                             f"{round(self.y_range_m[0], 3)}i"))
        top_right = ((f"{round(self.x_range_m[1], 3)} + "
                      f"{round(self.y_range_m[1], 3)}i")
                     if self.y_range_m[1] >= 0
                     else (f"{round(self.x_range_m[1], 3)} "
                           f"{round(self.y_range_m[1], 3)}i"))
        return f"Mandelbrot set of {func_name}, ({bottom_left}, {top_right})"

    def _title_generator_julia(self):
        func_name = ((f"z^2 + ({round(self.quadratic_map.c.real, 3)} + "
                      f"{round(self.quadratic_map.c.imag, 3)})i")
                     if self.quadratic_map.c.imag >= 0
                     else (f"z^2 + ({round(self.quadratic_map.c.real, 3)} "
                           f"{round(self.quadratic_map.c.imag, 3)})i"))
        bottom_left = ((f"{round(self.x_range_j[0], 3)} + "
                        f"{round(self.y_range_j[0], 3)}i")
                       if self.y_range_j[0] >= 0
                       else (f"{round(self.x_range_j[0], 3)} "
                             f"{round(self.y_range_j[0], 3)}i"))
        top_right = ((f"{round(self.x_range_j[1], 3)} + "
                      f"{round(self.y_range_j[1], 3)}i")
                     if self.y_range_j[1] >= 0
                     else (f"{round(self.x_range_j[1], 3)} "
                           f"{round(self.y_range_j[1], 3)}i"))
        return f"Julia set of {func_name}, ({bottom_left}, {top_right})"

    def _refresh_mandel(self):
        self.pil_img_mandel = self.quadratic_map.draw_mandelbrot(
            res_x=self.x_res_m,
            res_y=self.y_res_m,
            iterations=ITERATIONS,
            x_range=self.x_range_m,
            y_range=self.y_range_m,
            multiprocessing=self.multiprocessing)
        self._draw_external_rays(self.external_rays_angles)

    def _refresh_julia(self):
        self.pil_img_julia = self.quadratic_map.draw_julia(
            res_x=self.x_res_j,
            res_y=self.y_res_j,
            iterations=ITERATIONS,
            x_range=self.x_range_j,
            y_range=self.y_range_j,
            multiprocessing=self.multiprocessing)
        self._draw_external_rays_julia(self.external_rays_angles_julia)
        self._draw_equipotentials(self.equipotentials)

    def _click_event_mandel(self, event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.btn_down = True
            self.start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not self.drag:
            self.btn_down = False

        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            self.btn_down = False
            self.drag = False
            start_num = self._to_complex_m(*self.start_coords)
            end_num = self._to_complex_m(x, y)
            self.x_range_m = (min(start_num.real, end_num.real),
                              max(start_num.real, end_num.real))
            self.y_range_m = (min(start_num.imag, end_num.imag),
                              max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {self.x_range_m}, y: "
                  f"{self.y_range_m}...")
            ratio = ((self.x_range_m[1] - self.x_range_m[0])
                     / (self.y_range_m[1] - self.y_range_m[0])
                     if self.y_range_m[0] != self.y_range_m[1]
                     else 1)
            self.x_res_m = int(RESOLUTION*sqrt(ratio))
            self.y_res_m = int(RESOLUTION/sqrt(ratio))
            self._refresh_mandel()

        elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
            self.drag = True
            rectangle_open_cv_image_mandel = self.open_cv_image_mandel.copy()
            cv2.rectangle(rectangle_open_cv_image_mandel,
                          pt1=self.start_coords,
                          pt2=(x, y),
                          color=REC_COLOR,
                          thickness=2)
            cv2.imshow('mandel', rectangle_open_cv_image_mandel)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.quadratic_map.c = self._to_complex_m(x, y)
            print(f"Recalculating julia set with {self.quadratic_map.c} "
                  f"as c...")
            self._refresh_julia()

    def _click_event_julia(self, event, x, y, flags, params):
        """Process mouse interaction in julia set window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.btn_down = True
            self.start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not self.drag:
            self.btn_down = False
            self.quadratic_map.c = self._to_complex_j(*self.start_coords)
            print(f"Recalculating with {self.quadratic_map.c} as c...")
            self._refresh_julia()

        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            self.btn_down = False
            self.drag = False
            start_num = self._to_complex_j(*self.start_coords)
            end_num = self._to_complex_j(x, y)
            self.x_range_j = (min(start_num.real, end_num.real),
                              max(start_num.real, end_num.real))
            self.y_range_j = (min(start_num.imag, end_num.imag),
                              max(start_num.imag, end_num.imag))
            print(f"Recalculating in area x: {self.x_range_j}, y: "
                  f"{self.y_range_j}...")
            ratio = ((self.x_range_j[1] - self.x_range_j[0])
                     / (self.y_range_j[1] - self.y_range_j[0])
                     if self.y_range_j[0] != self.y_range_j[1]
                     else 1)
            self.x_res_j = int(RESOLUTION*sqrt(ratio))
            self.y_res_j = int(RESOLUTION/sqrt(ratio))
            self._refresh_julia()

        elif event == cv2.EVENT_MOUSEMOVE and self.btn_down:
            self.drag = True
            rectangle_open_cv_image_julia = self.open_cv_image_julia.copy()
            cv2.rectangle(rectangle_open_cv_image_julia,
                          pt1=self.start_coords,
                          pt2=(x, y),
                          color=REC_COLOR,
                          thickness=2)
            cv2.imshow('julia', rectangle_open_cv_image_julia)

    def _to_complex_m(self, x, y):
        x_val = self.x_range_m[0] + ((x / self.x_res_m)
                                     * (self.x_range_m[1] - self.x_range_m[0]))
        y_val = self.y_range_m[1] - ((y / self.y_res_m)
                                     * (self.y_range_m[1] - self.y_range_m[0]))
        return complex(x_val, y_val)

    def _to_complex_j(self, x, y):
        x_val = self.x_range_j[0] + ((x / self.x_res_j)
                                     * (self.x_range_j[1] - self.x_range_j[0]))
        y_val = self.y_range_j[1] - ((y / self.y_res_j)
                                     * (self.y_range_j[1] - self.y_range_j[0]))
        return complex(x_val, y_val)

    def _from_complex_m(self, z):
        x = ((z.real - self.x_range_m[0]) * self.x_res_m
             / (self.x_range_m[1] - self.x_range_m[0]))
        y = ((self.y_range_m[1] - z.imag) * self.y_res_m
             / (self.y_range_m[1] - self.y_range_m[0]))
        return int(x), int(y)

    def _from_complex_j(self, z):
        x = ((z.real - self.x_range_j[0]) * self.x_res_j
             / (self.x_range_j[1] - self.x_range_j[0]))
        y = ((self.y_range_j[1] - z.imag) * self.y_res_j
             / (self.y_range_j[1] - self.y_range_j[0]))
        return int(x), int(y)

    def _draw_external_rays(self, angles):
        for theta in angles:
            print(f"Drawing external ray at {theta}*2pi radians...")
            self.pil_img_mandel = self.quadratic_map.draw_ray_mandel(
                self.pil_img_mandel,
                res_x=self.x_res_m,
                res_y=self.y_res_m,
                x_range=self.x_range_m,
                y_range=self.y_range_m,
                theta=theta
            )

    def _draw_external_rays_julia(self, angles):
        angles = [2*pi*angle for angle in angles]
        for theta in angles:
            print(f"Drawing external ray at {theta} radians...")
            self.pil_img_julia = self.quadratic_map.draw_ray(
                self.pil_img_julia,
                res_x=self.x_res_j,
                res_y=self.y_res_j,
                x_range=self.x_range_j,
                y_range=self.y_range_j,
                angle=theta
            )

    def _draw_equipotentials(self, potentials):
        for potential in potentials:
            print(f"Drawing equipotential line at {potential}...")
            equipotential_im = self.quadratic_map.draw_eqpot(
                im=self.pil_img_julia,
                res_x=self.x_res_j,
                res_y=self.y_res_j,
                x_range=self.x_range_j,
                y_range=self.y_range_j,
                potential=potential
            )

    def _main_loop(self):
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('m'):
                print("Resetting Mandelbrot view...")
                self.x_res_m = RESOLUTION
                self.y_res_m = RESOLUTION
                self.x_range_m = X_RANGEM0
                self.y_range_m = Y_RANGEM0
                self._refresh_mandel()

            elif key == ord('j'):
                print("Resetting Julia view...")
                self.x_res_j = RESOLUTION
                self.y_res_j = RESOLUTION
                self.x_range_j = X_RANGEJ0
                self.y_range_j = Y_RANGEJ0
                self._refresh_julia()

            elif key == ord('r'):
                layout = [
                    [sg.Text('Would you like to draw external rays on the mand'
                             'elbrot or julia set?', size=(50, 2))],
                    [sg.Button('Mandelbrot', size=(15, 1)),
                     sg.Button('Julia', size=(15, 1)),
                     sg.Cancel(size=(15, 1))]
                ]
                window = sg.Window('External rays', layout)
                event, _ = window.read()
                window.close()
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    continue
                elif event == 'Mandelbrot':
                    layout = [
                        [sg.Text('Please enter the angle for the external ray '
                                 'as a multiple of 2pi (i.e. enter 1 to get 2p'
                                 'i radians).', size=(50, 2))],
                        [sg.Text('Theta', size=(10, 1)),
                         sg.InputText(size=(10, 1)),
                         sg.Button('Draw Ray', size=(25, 1))],
                        [sg.Text('Or enter the number of evenly-spaced rays yo'
                                 'u would like to draw.', size=(50, 2))],
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
                        self.external_rays_angles = []
                        self._refresh_mandel()
                    if event == 'Draw Ray':
                        try:
                            theta = float(values[0])
                        except(ValueError):
                            print("Not a valid angle. Angles must be a float.")
                            continue
                        self.external_rays_angles += [theta]
                        self._draw_external_rays([theta])
                    elif event == 'Draw Rays':
                        try:
                            count = int(values[1])
                        except(ValueError):
                            print("Not a valid number of rays. Number of rays "
                                  "must be an integer.")
                            continue
                        if count < 1:
                            print("Not a valid number of rays. Number of rays "
                                  "must be an integer.")
                            continue
                        theta_list = list(np.linspace(0, 1, count,
                                                      endpoint=False))
                        self.external_rays_angles += theta_list
                        self._draw_external_rays(theta_list)
                elif event == 'Julia':
                    layout = [
                        [sg.Text('Please enter the angle for the external ray '
                                 'as a multiple of 2pi (i.e. enter 1 to get 2p'
                                 'i radians).', size=(50, 2))],
                        [sg.Text('Theta', size=(10, 1)),
                         sg.InputText(size=(10, 1)),
                         sg.Button('Draw Ray', size=(25, 1))],
                        [sg.Text('Or enter the number of evenly-spaced rays yo'
                                 'u would like to draw.', size=(50, 2))],
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
                        self.external_rays_angles_julia = []
                        self._refresh_julia()
                    if event == 'Draw Ray':
                        try:
                            theta = float(values[0])
                        except(ValueError):
                            print("Not a valid angle. Angles must be a float.")
                            continue
                        self.external_rays_angles_julia += [theta]
                        self._draw_external_rays_julia([theta])
                    elif event == 'Draw Rays':
                        try:
                            count = int(values[1])
                        except(ValueError):
                            print("Not a valid number of rays. Number of rays "
                                  "must be an integer.")
                            continue
                        if count < 1:
                            print("Not a valid number of rays. Number of rays "
                                  "must be an integer.")
                            continue
                        theta_list = list(np.linspace(0, 1, count,
                                                      endpoint=False))
                        self.external_rays_angles_julia += theta_list
                        self._draw_external_rays_julia(theta_list)

            elif key == ord('e'):
                layout = [
                    [sg.Text('Please enter the potential for the equipotential'
                             ' line.', size=(50, 2))],
                    [sg.Text('Potential', size=(10, 1)),
                     sg.InputText(size=(10, 1)),
                     sg.Button('Draw Equipotential', size=(25, 1))],
                    [sg.Text('Or enter the number of evenly-logarithmically-sp'
                             'aced equipotential lines you would like to draw',
                             size=(50, 2))],
                    [sg.Text('Lines', size=(10, 1)),
                     sg.InputText(size=(10, 1)),
                     sg.Button('Draw Equipotentials', size=(25, 1))],
                    [sg.Button('Remove all equipotential lines', size=(22, 1)),
                     sg.Cancel(size=(23, 1))]
                ]
                window = sg.Window('Equipotential Lines', layout)
                event, values = window.read()
                window.close()
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    continue
                elif event == 'Remove all equipotential lines':
                    print("Removing equipotentials...")
                    self.equipotentials = []
                    self._refresh_julia()
                elif event == 'Draw Equipotential':
                    try:
                        potential = float(values[0])
                    except(ValueError):
                        print('Not a valid potential. Potentials must be a flo'
                              'at')
                        continue
                    self.equipotentials += [potential]
                    self._draw_equipotentials([potential])
                elif event == 'Draw Equipotentials':
                    try:
                        count = int(values[1])
                    except(ValueError):
                        print("Not a valid number of potentials. Number of pot"
                              "entials must be an integer.")
                        continue
                    if count < 1:
                        print("Not a valid number of potentials. Number of pot"
                              "entials must be positive.")
                        continue
                    potential_list = list(np.logspace(-5, 0, count, base=2))
                    self.equipotentials += potential_list
                    self._draw_equipotentials(potential_list)
