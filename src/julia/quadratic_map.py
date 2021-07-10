"""Module containing the QuadraticMap class."""
import numpy as np
import multiprocessing as mp
from functools import partial
from numba import jit
import cmath
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
from matplotlib import cm
from numbers import Number

from .map import Map, complex_to_pixel


class QuadraticMap(Map):
    """A quadratic mapping f: C -> C."""

    def __init__(self, c: float = None):
        """
        Construct an instance of the QuadraticMap class.

        A complex quadratic map p: C -> C of the form:
            p(z) = z^2 + c

        Parameters
        ----------
        c: float
            The term c in the quadratic map.
        """
        self.c = c
    
    @property
    def roots(self):
        if self.c == 0:
            return np.array([complex(0)])
        else:
            return np.array([cmath.sqrt(self.c)*1j, -cmath.sqrt(self.c)*1j])

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**2 + self.c

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 2*z

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(c, iterations, z_max):
        z = 0
        i = 0
        while i < iterations and abs(z) < z_max:
            z = z**2 + c
            i += 1
        return i / iterations

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, c, iterations, z_max):
        i = 0
        while i < iterations and abs(z) < z_max:
            z = z**2 + c
            i += 1
        return i / iterations

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-2, 2),
                              y_range: tuple = (-2, 2),
                              z_max: float = 2,
                              multiprocessing: bool = False) -> np.ndarray:
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_mandelbrot,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_mandelbrot,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-2, 2),
                         y_range: tuple = (-2, 2),
                         z_max: float = 2,
                         multiprocessing: bool = False) -> np.ndarray:
        if not isinstance(self.c, Number):
            raise ValueError(f'Expected Number for c, got {type(self.c)}.')
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_julia,
                                           c=self.c,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_julia,
                                      c=self.c,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def _calculate_ray_mandel(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              theta: float = 0,
                              D: float = 50,
                              S: float = 20,
                              R: float =200,
                              error: float = 0.1):
        """
        Construct an array of points on the external ray of angle theta.

        Parameters
        ----------
        theta: float
            angle of the external ray
        D: int
            depth of the ray
        S: int
            sharpness of the ray
        R: int
            radius
        error: float
            error used for convergence of newton method
        """
        points = [R * cmath.exp(2 * np.pi * theta * 1j)]
        for i in range(1, D + 1):
            for q in range(1, S + 1):
                r_m = R ** (1 / (2 ** (i - 1 + q / S)))
                t_m = r_m**(2**(i)) * cmath.exp(2 * np.pi * 1j * theta * 2**(i))
                c_next = points[-1]
                c_previous = 0
                while abs(c_previous - c_next) >= error:
                    C_k = c_next
                    D_k = 1
                    for _ in range(i):
                        D_k = 2 * D_k * C_k + 1
                        C_k = C_k ** 2 + c_next
                    c_previous = c_next
                    c_next = c_previous - (C_k - t_m) / D_k
                points.append(c_next)
        points = list(filter(lambda x: abs(x.real) < 3 and abs(x.imag) < 3, points))
        points = [complex_to_pixel(point, res_x=res_x, res_y=res_y, x_range=x_range, y_range=y_range) for point in points]
        return points

    def draw_ray_mandel(self,
                        im: Image = None,
                        res_x: int = 600,
                        res_y: int = 600,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3),
                        line_weight: int = 1,
                        **kwargs):
        if im is None:
            im = self.draw_mandelbrot(res_x=res_x,
                                      res_y=res_y,
                                      x_range=x_range,
                                      y_range=y_range)
        else:
            res_x, res_y = im.size
        d = ImageDraw.Draw(im)
        ray = self._calculate_ray_mandel(**kwargs)
        d.line(ray, fill=(0, 0, 0),
               width=line_weight, joint="curve")
        return im
    
    def _composition(self, z, iters):
        result = z
        for i in range(iters):
            result = self.__call__(result)
        return result

    def _newton_map_julia(self, z, n, R, theta, error=0.001):
        new_result = z
        old_result = 0
        C_k = z
        D_k = 1
        for j in range(n):
            D_k = 2*D_k*C_k
            C_k = C_k**2 + self.c
        while abs(old_result - new_result) >= error:
            old_result = new_result
            new_result = old_result - (C_k - R*cmath.exp(2 * np.pi * 1j * theta * 2**n))/D_k 
            #print(new_result)
        return new_result

    def _newton_map_julia_log(self, z, n, R, theta, error=0.001):
        new_result = z
        old_result = 0
        while abs(old_result - new_result) >= error:
            old_result = new_result
            fn_prime = 2 * old_result
            for j in range(2, n+1):  
                fn_prime *= 2 * self.composition(old_result, j)
            #print(fn_prime,self.composition(old_result, n), R*cmath.exp(2 * np.pi * 1j * theta * 2**n))
            new_result = old_result - (((cmath.log(self.composition(old_result, n)) - cmath.log(R) - 1j * np.pi * theta))*self.composition(old_result,n))/(fn_prime)
            new_result = cmath.log(new_result) 
        return new_result 

    @staticmethod
    @jit(nopython=True)
    def _q(z, c):
        return 1 + c/(z**2)

    @staticmethod
    @jit(nopython=True)
    def _dq(z, c):
        return -2*c/(z**3)

    @staticmethod
    @jit(nopython=True)
    def _f(z, c):
        return z**2+c

    @ staticmethod
    @ jit(nopython=True)
    def _df(z):
        return 2*z

    @ staticmethod
    @ jit(nopython=False) 
    def _phi_newton(w_list, c, f, df, q, dq, phi_iters, newt_iters):
        z = w_list[0]
        z_list = []
        for w in w_list:
            for i in range(newt_iters):
                phi = z * q(z, c)**(1.0/2.0) 
                dphi = 1/z + dq(z, c)/(2*q(z, c))
                prev_f = z
                prev_df = complex(1)
                for k in range(2, phi_iters):
                    prev_df *= df(prev_f)
                    prev_f = f(prev_f, c)
                    factor = q(prev_f, c)**(2.0**-k)
                    summand = ((2.0**-k)*dq(prev_f, c)
                               / q(prev_f, c)*prev_df)
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
                                       self.c,
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

    @staticmethod
    @jit(nopython=True)
    def _f(z, c):
        return z**2 + c
        
    @staticmethod
    @jit(nopython=True)
    def _bottcher(f, z, c, max_n=5):
        total = 1
        for n in range(1, max_n + 1):
            f_n = z
            for i in range(n):
                f_n = f(f_n, c)
            total *= (f_n/(f_n - c))**(1/(2**n))
        total = z * total
        return total
    
    @staticmethod
    @jit(nopython=True)
    def _potential(f, bottcher, z, c, max_n=5):
        return math.log(abs(bottcher(f, z, c, max_n)))
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_eqpot(f, bottcher, potential, c, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), max_n=5):
        results = np.zeros((res_x, res_y))
        step_x = abs((x_range[1] - x_range[0])/res_x)
        step_y = abs((y_range[1] - y_range[0])/res_y)
        for x_i, x in enumerate(np.linspace(x_range[0], x_range[1], res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0], y_range[1], res_y)):
                c1 = complex(x, y)
                c2 = complex(x + step_x, y)
                c3 = complex(x, y + step_y)
                c4 = complex(x + step_x, y + step_y)
                pot1 = potential(f, bottcher, c1, c, max_n)
                pot2 = potential(f, bottcher, c2, c, max_n)
                pot3 = potential(f, bottcher, c3, c, max_n)
                pot4 = potential(f, bottcher, c4, c, max_n)
                if min(pot1, pot2, pot3, pot4) <= equipotential <= max(pot1, pot2, pot3, pot4) :
                    results[x_i, y_i] = 1
        return results
    
    def draw_eqpot(self,
                   im: Image = None,
                   res_x: int =600,
                   res_y: int =600,
                   x_range: tuple =(-3, 3),
                   y_range: tuple =(-3, 3),
                   max_n: int =5,
                   potential: float = 1.,
                   line_weight: int = 1) -> Image.Image:
        if im is None:
            im = self.draw_julia(res_x=res_x,
                                 res_y=res_y,
                                 x_range=x_range,
                                 y_range=y_range)
        else:
            res_x, res_y = im.size
        eqpot = self._calculate_eqpot(self._f,
                                       self._bottcher,
                                       self._potential,
                                       self.c,
                                       potential,
                                       res_x, res_y,
                                       x_range, y_range,
                                       max_n)
        eqpot = np.rot90(eqpot)
        eqpot_im = Image.fromarray(np.uint8(cm.cubehelix_r(eqpot)*255)).convert("RGBA")
        im_data = eqpot_im.getdata()

        trans_im_data = []
        for item in im_data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                trans_im_data.append((255, 255, 255, 0))
            else:
                trans_im_data.append(item)
        
        eqpot_im.putdata(trans_im_data)
        im.paste(eqpot_im, (0, 0), eqpot_im)
        return im


class QuadraticNewtonMap(Map):
    """A Newton mapping f: C -> C, i.e. f(z) = z - g'(z)/g(z)."""

    def __init__(self, quadratic: QuadraticMap):
        """
        Construct an instance of the QuadraticNewtonMap class.

        Parameters
        ----------
        quadratic: QuadraticNewtonMap
            The quadratic to find the Newton map for.
        """
        self.quadratic = quadratic

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z - self.quadratic(z)/self.quadratic.derivative(z)

    @staticmethod
    @jit(nopython=True)
    def _conv_time_julia(z, c, roots, iterations, tol):
        result = [0, 0, 0]
        for i in range(iterations):
            z = z/2 if c == 0 else z - (z**2 + c)/(2*z)
            for j, r in enumerate(roots):
                if abs(z-r) < tol:
                    result[j] = int(255*(1-i/iterations))
                    return result
        return result

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         tol: float = 1e-12,
                         multiprocessing: bool = False) -> np.ndarray:
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        for i, r in enumerate(self.quadratic.roots):
            for r_ in self.quadratic.roots[i+1:]:
                tol = min(tol, abs(r-r_)/3)
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._conv_time_julia,
                                           c=self.quadratic.c,
                                           roots=self.quadratic.roots,
                                           iterations=iterations,
                                           tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        else:
            result_list = map(partial(self._conv_time_julia,
                                      c=self.quadratic.c,
                                      roots=self.quadratic.roots,
                                      iterations=iterations,
                                      tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        return results

    @staticmethod
    @jit(nopython=True)
    def _phi_inv(w_list, roots):
        z_list = np.zeros((w_list.shape[0]*len(roots), w_list.shape[1]),
                          dtype=np.cdouble)
        for root_idx, r in enumerate(roots):
            for m, row in enumerate(w_list):
                for n, w in enumerate(row):
                    z_list[len(roots)*m+root_idx, n] = r*(w+1)/(w-1)
        return z_list

    def _calculate_ray(self,
                       res_x: int = 600,
                       res_y: int = 600,
                       x_range: tuple = (-3, 3),
                       y_range: tuple = (-3, 3),
                       angles: list = [0.],
                       res_ray: int = 1024):
        w_list = np.array([[cmath.rect(1/np.sin(r), angle) for r in
                          np.linspace(0, np.pi/2, res_ray+2)[1:-1]]
                          for angle in angles])
        result_list = self._phi_inv(w_list,
                                    self.quadratic.roots)
        return map(list, list(map(partial(map, partial(complex_to_pixel,
                                                       res_x=res_x,
                                                       res_y=res_y,
                                                       x_range=x_range,
                                                       y_range=y_range)),
                                  result_list)))

    def draw_ray(self,
                 im: Image = None,
                 res_x: int = 600,
                 res_y: int = 600,
                 x_range: tuple = (-3, 3),
                 y_range: tuple = (-3, 3),
                 angles: list = [0.],
                 res_ray: int = 1024,
                 line_weight: int = 1) -> Image.Image:
        """
        Draw internal rays of the specified angle at all roots.

        Parameters
        ----------
        im: Image
            The image on which to overla the rays. If None, the Julia set is
            used.
        res_x: int
            The horizontal resolution of the image.
        res_y: int
            The vertical resolution of the image.
        x_range: (float, float)
            The range of x values to consider.
        y_range: (float, float)
            The range of y values to consider.
        angles: list
            The angles of the internal rays to be drawn.
        res_ray: float
            The resolution of the ray.
        line_weight: int
            The pixel width of the ray.

        Returns
        -------
        im: Image.Image
            The image of the internal rays as a Pillow image object.
        """
        if im is None:
            im = self.draw_julia(res_x=res_x,
                                 res_y=res_y,
                                 x_range=x_range,
                                 y_range=y_range)
        else:
            res_x, res_y = im.size
        d = ImageDraw.Draw(im)
        rays = self._calculate_ray(res_x=res_x,
                                   res_y=res_y,
                                   x_range=x_range,
                                   y_range=y_range,
                                   angles=angles,
                                   res_ray=res_ray)
        for ray in rays:
            d.line(ray, fill=(255, 255, 255),
                   width=line_weight, joint="curve")
        return im

    def _calculate_eqpot(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         potentials: float = 1.0,
                         res_eqpot: int = 1024):
        w_list = np.array([[cmath.rect(np.exp(potential), angle) for angle in
                           np.linspace(-np.pi, np.pi, res_eqpot+1)[:-1]]
                          for potential in potentials])
        result_list = self._phi_inv(w_list,
                                    self.quadratic.roots)
        return map(list, list(map(partial(map, partial(complex_to_pixel,
                                                       res_x=res_x,
                                                       res_y=res_y,
                                                       x_range=x_range,
                                                       y_range=y_range)),
                                  result_list)))

    def draw_eqpot(self,
                   im: Image = None,
                   res_x: int = 600,
                   res_y: int = 600,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   potentials: list = [1.],
                   line_weight: int = 1) -> Image.Image:
        """
         Draw equipotential lines of the specified potential at all roots.

         Parameters
         ----------
         im: Image
             The image on which to overla the equipotentials. If None, the
             Julia set is used.
         res_x: int
             The horizontal resolution of the image.
         res_y: int
             The vertical resolution of the image.
         x_range: (float, float)
             The range of x values to consider.
         y_range: (float, float)
             The range of y values to consider.
         potentials: list
             The potential of the line.
         res_eqpot: float
             The resolution of the equipotential line.
         line_weight: int
             The pixel width of the equipotental line.

         Returns
         -------
         im: Image.Image
             The image of the equipotential line as a Pillow image object.
         """
        if im is None:
            im = self.draw_julia(res_x=res_x,
                                 res_y=res_y,
                                 x_range=x_range,
                                 y_range=y_range)
        else:
            res_x, res_y = im.size
        d = ImageDraw.Draw(im)
        eqpots = self._calculate_eqpot(res_x=res_x,
                                       res_y=res_y,
                                       x_range=x_range,
                                       y_range=y_range,
                                       potentials=potentials)
        for eqpot in eqpots:
            d.line(eqpot, fill=(255, 255, 255),
                   width=line_weight, joint="curve")
        return im
