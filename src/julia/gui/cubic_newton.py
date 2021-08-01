"""Module containing the CubicWindows class."""
import cv2
import numpy as np
from math import sqrt, pi
import PySimpleGUI as sg

from julia.cubic_map import CubicMap, CubicNewtonMap

from .constants import (X_RANGEM0, Y_RANGEM0, X_RANGEJ0, Y_RANGEJ0, RESOLUTION,
                        ITERATIONS, REC_COLOR)


class CubicNewtonWindows:
    """The class for the cubic GUI."""

    def __init__(self, multiprocessing):
        self.multiprocessing = multiprocessing

        self.btn_down, self.drag = False, False
        self.x_range_m, self.y_range_m = X_RANGEM0, Y_RANGEM0
        self.x_range_j, self.y_range_j = X_RANGEM0, Y_RANGEM0

        self.start_coords = None
        self.x_res_m, self.y_res_m = RESOLUTION, RESOLUTION
        self.x_res_j, self.y_res_j = RESOLUTION, RESOLUTION

        self.internal_rays_angles = []
        self.equipotentials = []

        cubic = CubicMap(a=0, b=1)
        self.cubic_newton = CubicNewtonMap(cubic)

    def start(self):
        """Start the cubic GUI."""
        root = sg.tk.Tk()  # DO NOT DELETE LINES 33-44 OR STUFF BREAKS
        root.withdraw()

        cv2.namedWindow('Loading...')
        cv2.setWindowProperty("Loading...",
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        cv2.setWindowProperty("Loading...",
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Loading...")

        sg.SetOptions(font='Helvetica 15', border_width=5)
        sg.theme('Material1')

        self._refresh_julia()

        cv2.moveWindow('julia', 0, 0)
        cv2.setMouseCallback('julia', self._click_event_julia)
        self._main_loop()

    def _title_generator(self):
        func_name = ((f"z - (z^3 - ({round(self.cubic_newton.cubic.a.real, 3)}"
                      f" + {round(self.cubic_newton.cubic.a.imag, 3)}i)z + b)/"
                      f"(3z^2 - ({round(self.cubic_newton.cubic.a.real, 3)} + "
                      f"{round(self.cubic_newton.cubic.a.imag, 3)}i))")
                     if self.cubic_newton.cubic.a.imag >= 0
                     else
                     (f"z - (z^3 - ({round(self.cubic_newton.cubic.a.real, 3)}"
                      f" {round(self.cubic_newton.cubic.a.imag, 3)}i)z + b)/(3"
                      f"z^2 - ({round(self.cubic_newton.cubic.a.real, 3)} "
                      f"{round(self.cubic_newton.cubic.a.imag, 3)}i))"))
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
        a_repr = ((f"({round(self.cubic_newton.cubic.a.real, 3)} + "
                   f"{round(self.cubic_newton.cubic.a.imag, 3)}i)")
                  if self.cubic_newton.cubic.a.imag >= 0
                  else (f"({round(self.cubic_newton.cubic.a.real, 3)} "
                        f"{round(self.cubic_newton.cubic.a.imag, 3)}i)"))
        b_repr = ((f"({round(self.cubic_newton.cubic.b.real, 3)} + "
                   f"{round(self.cubic_newton.cubic.b.imag, 3)}i)")
                  if self.cubic_newton.cubic.b.imag >= 0
                  else (f"({round(self.cubic_newton.cubic.b.real, 3)} "
                        f"{round(self.cubic_newton.cubic.b.imag, 3)}i)"))

        func_name = f"z - (z^3 - {a_repr}z + {b_repr})/(3z^2 - {a_repr})"
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

    def _refresh_julia(self):
        self.pil_img_julia = self.cubic_newton.draw_julia(
            res_x=self.x_res_j,
            res_y=self.y_res_j,
            iterations=ITERATIONS,
            x_range=self.x_range_j,
            y_range=self.y_range_j,
            multiprocessing=self.multiprocessing)
        self.open_cv_image_julia = np.array(
            self.pil_img_julia.convert('RGB'))[:, :, ::-1]
        cv2.imshow('julia', self.open_cv_image_julia)
        cv2.setWindowTitle('julia', self._title_generator_julia())
        self._draw_internal_rays(self.internal_rays_angles)
        self._draw_equipotentials(self.equipotentials)

    def _click_event_julia(self, event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.btn_down = True
            self.start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not self.drag:
            self.btn_down = False
            self.cubic_newton.cubic.a = self._to_complex_j(*self.start_coords)
            print(f"Recalculating with {self.cubic_newton.cubic.a} as a...")
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

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cubic_newton.cubic.b = self._to_complex_j(x, y)
            print(f"Recalculating julia set with {self.cubic_newton.cubic.b} a"
                  f"s b...")
            self._refresh_julia()

    def _to_complex_j(self, x, y):
        x_val = self.x_range_j[0] + ((x / self.x_res_j)
                                     * (self.x_range_j[1] - self.x_range_j[0]))
        y_val = self.y_range_j[1] - ((y / self.y_res_j)
                                     * (self.y_range_j[1] - self.y_range_j[0]))
        return complex(x_val, y_val)

    def _from_complex_j(self, z):
        x = ((z.real - self.x_range_j[0]) * self.x_res_j
             / (self.x_range_j[1] - self.x_range_j[0]))
        y = ((self.y_range_j[1] - z.imag) * self.y_res_j
             / (self.y_range_j[1] - self.y_range_j[0]))
        return int(x), int(y)

    def _draw_internal_rays(self, angles):
        angles = [2*pi*theta for theta in angles]
        if angles:
            print("Drawing internal rays...")
            self.pil_img_julia = self.cubic_newton.draw_ray(
                im=self.pil_img_julia,
                res_x=self.x_res_j,
                res_y=self.y_res_j,
                x_range=self.x_range_j,
                y_range=self.y_range_j,
                angles=angles,
            )
            self.open_cv_image_julia = np.array(
                self.pil_img_julia.convert('RGB'))[:, :, ::-1]
            cv2.imshow('julia', self.open_cv_image_julia)

    def _draw_equipotentials(self, potentials):
        if potentials:
            print("Drawing equipotentials...")
            self.pil_img_julia = self.cubic_newton.draw_eqpot(
                im=self.pil_img_julia,
                res_x=self.x_res_j,
                res_y=self.y_res_j,
                x_range=self.x_range_j,
                y_range=self.y_range_j,
                potentials=potentials
            )
            self.open_cv_image_julia = np.array(
                self.pil_img_julia.convert('RGB'))[:, :, ::-1]
            cv2.imshow('julia', self.open_cv_image_julia)

    def _main_loop(self):
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            elif key == ord('j'):
                print("Resetting Julia view...")
                self.x_res_j = RESOLUTION
                self.y_res_j = RESOLUTION
                self.x_range_j = X_RANGEJ0
                self.y_range_j = Y_RANGEJ0
                self._refresh_julia()

            elif key == ord('r'):
                layout = [
                    [sg.Text('Please enter the angle for the internal ray as a'
                             ' multiple of 2pi (i.e. enter 1 to get 2pi radian'
                             's).', size=(50, 2))],
                    [sg.Text('Theta', size=(10, 1)),
                     sg.InputText(size=(10, 1)),
                     sg.Button('Draw Ray', size=(25, 1))],
                    [sg.Text('Or enter the number of evenly-spaced rays you wo'
                             'uld like to draw.', size=(50, 2))],
                    [sg.Text('Rays', size=(10, 1)), sg.InputText(size=(10, 1)),
                     sg.Button('Draw Rays', size=(25, 1))],
                    [sg.Button('Remove all internal rays', size=(22, 1)),
                     sg.Cancel(size=(23, 1))]
                ]
                window = sg.Window('Internal rays', layout)
                event, values = window.read()
                window.close()
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    continue
                elif event == 'Remove all internal rays':
                    print("Removing internal rays...")
                    self.internal_rays_angles = []
                    self._refresh_julia()
                if event == 'Draw Ray':
                    try:
                        theta = float(values[0])
                    except(ValueError):
                        print("Not a valid angle. Angles must be a float.")
                        continue
                    self.internal_rays_angles += [theta]
                    self._draw_internal_rays([theta])
                elif event == 'Draw Rays':
                    try:
                        count = int(values[1])
                    except(ValueError):
                        print("Not a valid number of rays. Number of rays must"
                              " be an integer.")
                        continue
                    theta_list = list(np.linspace(0, 1, count, endpoint=False))
                    self.internal_rays_angles += theta_list
                    self._draw_internal_rays(theta_list)

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
                    potential_list = list(np.logspace(-5, 3, count, base=2))
                    self.equipotentials += potential_list
                    self._draw_equipotentials(potential_list)
