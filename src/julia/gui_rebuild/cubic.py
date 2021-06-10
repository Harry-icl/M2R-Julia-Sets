"""Module containing the CubicWindows class."""
import cv2
import numpy as np
from math import sqrt
import PySimpleGUI as sg

from julia.cubic_map import CubicMap

from .constants import (X_RANGEM0, Y_RANGEM0, X_RANGEJ0, Y_RANGEJ0, RESOLUTION,
                        ITERATIONS, REC_COLOR, RAY_COLOR)


class CubicWindows:
    """The class for the cubic GUI."""

    def __init__(self, multiprocessing):
        self.multiprocessing = multiprocessing

        self.btn_down, self.drag = False, False
        self.x_range_m, self.y_range_m = X_RANGEM0, Y_RANGEM0
        self.x_range_j, self.y_range_j = X_RANGEM0, Y_RANGEM0

        self.start_coords = None
        self.x_res_m, self.y_res_m = RESOLUTION, RESOLUTION
        self.x_res_j, self.y_res_j = RESOLUTION, RESOLUTION

        self.external_rays_angles = []

        self.cubic_map = CubicMap(a=0, b=0)

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

        self._refresh_mandel()
        self._refresh_julia()

        cv2.moveWindow('mandel', 0, 0)
        cv2.moveWindow('julia', RESOLUTION, 0)
        cv2.setMouseCallback('mandel', self._click_event_mandel)
        cv2.setMouseCallback('julia', self._click_event_julia)
        self._main_loop()

    def _title_generator(self):
        func_name = ((f"z^3 - ({round(self.cubic_map.a.real, 3)} + "
                      f"{round(self.cubic_map.a.imag, 3)}i)z + b")
                     if self.cubic_map.a.imag >= 0
                     else
                     (f"z^3 - ({round(self.cubic_map.a.real, 3)} "
                      f"{round(self.cubic_map.a.imag, 3)}i)z + b"))
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
        a_repr = ((f"({round(self.cubic_map.a.real, 3)} + "
                   f"{round(self.cubic_map.a.imag, 3)}i)")
                  if self.cubic_map.a.imag >= 0
                  else (f"({round(self.cubic_map.a.real, 3)} "
                        f"{round(self.cubic_map.a.imag, 3)}i)"))
        b_repr = ((f"({round(self.cubic_map.b.real, 3)} + "
                   f"{round(self.cubic_map.b.imag, 3)}i)")
                  if self.cubic_map.b.imag >= 0
                  else (f"({round(self.cubic_map.b.real, 3)} "
                        f"{round(self.cubic_map.b.imag, 3)}i)"))

        func_name = f"z^3 - {a_repr}z + {b_repr}"
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
        self.pil_img_mandel = self.cubic_map.draw_mandelbrot(
            res_x=self.x_res_m,
            res_y=self.y_res_m,
            iterations=ITERATIONS,
            x_range=self.x_range_m,
            y_range=self.y_range_m,
            multiprocessing=self.multiprocessing)
        self.open_cv_image_mandel = np.array(
            self.pil_img_mandel.convert('RGB'))
        cv2.imshow('mandel', self.open_cv_image_mandel)
        cv2.setWindowTitle('mandel', self._title_generator())
        self._draw_external_rays(self.external_rays_angles)

    def _refresh_julia(self):
        self.pil_img_julia = self.cubic_map.draw_julia(
            res_x=self.x_res_j,
            res_y=self.y_res_j,
            iterations=ITERATIONS,
            x_range=self.x_range_j,
            y_range=self.y_range_j,
            multiprocessing=self.multiprocessing)
        self.open_cv_image_julia = np.array(self.pil_img_julia.convert('RGB'))
        cv2.imshow('julia', self.open_cv_image_julia)
        cv2.setWindowTitle('julia', self._title_generator_julia())

    def _click_event_mandel(self, event, x, y, flags, params):
        """Process mouse interaction via cv2."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.btn_down = True
            self.start_coords = (x, y)
            cv2.waitKey(10)  # this needs to be here so that clicks are \
            # registered as such, otherwise a tiny drag will be detected.

        elif event == cv2.EVENT_LBUTTONUP and not self.drag:
            self.btn_down = False
            self.cubic_map.a = self._to_complex_m(*self.start_coords)
            print(f"Recalculating with {self.cubic_map.a} as a...")
            self._refresh_mandel()

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
            self.cubic_map.b = self._to_complex_m(x, y)
            print(f"Recalculating julia set with {self.cubic_map.b} as b...")
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
            self.cubic_map.b = self._to_complex_j(*self.start_coords)
            print(f"Recalculating with {self.cubic_map.b} as b...")
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
            ray = [self._from_complex_m(z)
                   for z in self.cubic_map.external_ray(theta)]
            pairs = zip(ray[:-1], ray[1:])

            for pair in pairs:
                cv2.line(self.open_cv_image_mandel,
                         pair[0], pair[1],
                         color=RAY_COLOR, thickness=1)
            cv2.imshow('mandel', self.open_cv_image_mandel)

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

            elif key == ord('e'):
                sg.theme('Material1')
                layout = [
                    [sg.Text('Please enter the angle for the external ray as a'
                             ' multiple of 2pi (i.e. enter 1 to get 2pi radian'
                             's).', size=(50, 2))],
                    [sg.Text('Theta', size=(10, 1)),
                     sg.InputText(size=(10, 1)),
                     sg.Button('Draw Ray', size=(25, 1))],
                    [sg.Text('Or enter the number of evenly-spaced rays you wo'
                             'uld like to draw.', size=(50, 2))],
                    [sg.Text('Rays', size=(10, 1)), sg.InputText(size=(10, 1)),
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
                        print("Not a valid number of rays. Number of rays must"
                              " be an integer.")
                        continue
                    theta_list = list(np.linspace(0, 1, count))
                    self.external_rays_angles += theta_list
                    self._draw_external_rays(theta_list)
