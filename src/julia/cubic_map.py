import cmath
from functools import partial
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit
import math
from PIL import Image, ImageDraw
from matplotlib import cm
from .map import Map, complex_to_pixel
class CubicMap(Map):
    """A cubic mapping f: C -> C."""

    def __init__(self, a: float = None, b: float = None):
        """
        Construct an instance of the CubicMap class.

        A complex cubic map p: C -> C of the form:
            p(z) = z^3 - az + b

        Parameters
        ----------
        a: float
            The term a in the cubic map.
        b: float
            The term b in the cubic map.
        """
        self.a = a
        self.b = b

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**3 - self.a*z + self.b

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 3*z**2 - self.a

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(b, a, c1, c2, iterations, z_max):
        z1 = c1
        z2 = c2
        z1_diverge = False
        z2_diverge = False
        for i in range(iterations):
            z1 = z1**3 - a*z1 + b if not z1_diverge else z1
            z2 = z2**3 - a*z2 + b if not z2_diverge else z2
            if abs(z1 - c1) > z_max:
                z1_diverge = True
            if abs(z2 - c2) > z_max:
                z2_diverge = True
            if z1_diverge and z2_diverge:
                return i / iterations
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, a, b, iterations, z_max):
        for i in range(iterations):
            z = z**3 - a*z + b
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False) -> np.ndarray:
        c1 = -cmath.sqrt(self.a/3)
        c2 = cmath.sqrt(self.a/3)
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_mandelbrot,
                                           a=self.a,
                                           c1=c1,
                                           c2=c2,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_mandelbrot,
                                      a=self.a,
                                      c1=c1,
                                      c2=c2,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3,
                         multiprocessing: bool = False) -> np.ndarray:
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_julia,
                                           a=self.a,
                                           b=self.b,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_julia,
                                      a=self.a,
                                      b=self.b,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results
    
    @staticmethod
    @jit(nopython=True)
    def _q(z, a, b):
        return 1 - a/(z**2) + b/(z**3)

    @staticmethod
    @jit(nopython=True)
    def _dq(z, a, b):
        return 2*a/(z**3) - 3*b/(z**4)

    @staticmethod
    @jit(nopython=True)
    def _f(z, a, b):
        return z**3 - a*z + b

    @ staticmethod
    @ jit(nopython=True)
    def _df(z, a):
        return 3*z**2 - a

    @ staticmethod
    @ jit(nopython=False) 
    def _phi_newton(w_list, a, b, f, df, q, dq, phi_iters, newt_iters):
        z = w_list[0]
        z_list = []
        for w in w_list:
            for i in range(newt_iters):
                phi = z * q(z, a, b)**(1.0/3.0) 
                dphi = 1/z + dq(z, a, b)/(3*q(z, a, b))
                prev_f = z
                prev_df = complex(1)
                for k in range(2, phi_iters):
                    prev_df *= df(prev_f, a)
                    prev_f = f(prev_f, a, b)
                    factor = q(prev_f, a, b)**(3.0**-k)
                    summand = ((3.0**-k)*dq(prev_f, a, b)
                               / q(prev_f, a, b)*prev_df)
                    if not (cmath.isnan(factor) or cmath.isnan(summand)):
                        phi *= factor
                        dphi += summand
                    elif not cmath.isnan(factor):
                        phi *= factor
                    elif not cmath.isnan(summand):
                        dphi += summand
                    else:
                        break
                z = z-1/dphi*(1-w/phi)
            z_list.append(z)
        return z_list

    def _calculate_ray(self,
                       res_x: int = 600,
                       res_y: int = 600,
                       x_range: tuple = (-3, 3),
                       y_range: tuple = (-3, 3),
                       angle: float = 0,
                       res_ray: int = 2048,
                       phi_iters: int = 128,
                       newt_iters: int = 256):
        w_list = np.array([cmath.rect(1/np.sin(r), angle) for r in
                          np.linspace(0, np.pi/2, res_ray+2)[1:-1]])
        result_list = self._phi_newton(w_list,
                                       self.a,
                                       self.b,
                                       self._f,
                                       self._df,
                                       self._q,
                                       self._dq,
                                       phi_iters,
                                       newt_iters)
        return list(map(partial(complex_to_pixel,
                                res_x=res_x,
                                res_y=res_y,
                                x_range=x_range,
                                y_range=y_range),
                        result_list))

    def draw_ray(self,
                 im: Image = None,
                 res_x: int = 600,
                 res_y: int = 600,
                 x_range: tuple = (-3, 3),
                 y_range: tuple = (-3, 3),
                 angle: float = 0,
                 res_ray: int = 1024,
                 phi_iters: int = 128,
                 newt_iters: int = 256,
                 line_weight: int = 1):
        if im is None:
            im = self.draw_julia(res_x=res_x,
                                 res_y=res_y,
                                 x_range=x_range,
                                 y_range=y_range)
        else:
            res_x, res_y = im.size
        d = ImageDraw.Draw(im)
        ray = self._calculate_ray(res_x=res_x,
                                    res_y=res_y,
                                    x_range=x_range,
                                    y_range=y_range,
                                    angle=angle,
                                    res_ray=res_ray,
                                    phi_iters=phi_iters,
                                    newt_iters=newt_iters)
        d.line(ray, fill=(0, 0, 0),
                width=line_weight, joint="curve")
        return im



class CubicNewtonMap(Map):
    """A Newton map f(z) = z - g'(z)/g(z) where g is cubic."""

    def __init__(self, cubic: CubicMap):
        """
        Construct an instance of the CubicNewtonMap class.

        Parameters
        ----------
        cubic: CubicNewtonMap
            The cubic to find the Newton map for.
        """
        self.cubic = cubic

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z - self.cubic.derivative(z)/self.cubic(z)

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False):
        raise NotImplementedError

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3,
                         multiprocessing: bool = False):
        raise NotImplementedError
