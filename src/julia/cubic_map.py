"""Module containing map classes for cubic functions."""
import cmath
from functools import partial
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit
import math
from matplotlib import cm
from numbers import Number
from PIL import Image, ImageDraw

from .map import Map, complex_to_pixel


class CubicMap(Map):
    """A cubic mapping f: C -> C."""

    def __init__(self, a: float = 0, b: float = 0):
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
    
    @property
    def roots(self):
        if self.a == self.b == 0:
            return np.array([0])
        elif self.b == 0:
            return np.array([complex(0), cmath.sqrt(self.a), -cmath.sqrt(self.a)])
        elif self.a == 0:
            gamma = complex((-self.b)**(1/3))
        else:
            gamma = (-self.b/2+cmath.sqrt(self.b**2/4-self.a**3/27))**(1/3)
        omega = cmath.rect(1, 2*np.pi/3)
        omega_ = cmath.rect(1, -2*np.pi/3)
        cub_roots = [gamma + self.a/(3*gamma)]
        if np.all(~np.isclose(gamma*omega + self.a/(3*gamma)*omega_, cub_roots)):
            cub_roots.append(gamma*omega + self.a/(3*gamma)*omega_)
        if np.all(~np.isclose(gamma*omega_ + self.a/(3*gamma)*omega, cub_roots)):
            cub_roots.append(gamma*omega_ + self.a/(3*gamma)*omega)
        cub_roots = np.array(cub_roots)
        return cub_roots

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**3 - self.a*z + self.b

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 3*z**2 - self.a

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(b, a, c1, c2, iterations, z_max):
        z1 = c1
        z2 = c2
        i = 0
        while i < iterations and abs(z1 - c1) < z_max and abs(z2 - c2) < z_max:
            z1 = z1**3 - a*z1 + b
            z2 = z2**3 - a*z2 + b
            i += 1
        return i / iterations

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, a, b, iterations, z_max):
        i = 0
        while i < iterations and abs(z) < z_max:
            z = z**3 - a*z + b
            i += 1
        return i / iterations

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False) -> np.ndarray:
        if not isinstance(self.a, Number):
            raise ValueError(f"Expected Number for a, got {type(self.a)}.")
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
        if not isinstance(self.a, Number):
            raise ValueError(f"Expected Number for a, got {type(self.a)}.")
        if not isinstance(self.b, Number):
            raise ValueError(f"Expected Number for b, got {type(self.b)}.")
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
    
    def _calculate_ray_mandel(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              theta: float = 0,
                              D: float = 20,
                              S: float = 10,
                              R: float = 50,
                              error: float = 0.01):
        points = [R * cmath.exp(2 * np.pi * theta * 1j)]
        for i in range(1, D + 1):
            for q in range(1, S + 1):
                r_m = R ** (1 / (2 ** (i - 1 + q / S)))
                t_m = r_m**(2**(i)) * cmath.exp(2 * np.pi * 1j * theta * 3**(i))
                b_next = points[-1]
                b_previous = 0
                while abs(b_previous - b_next) >= error:
                    C_k = b_next
                    D_k = 1
                    for _ in range(i):
                        D_k = 3 * D_k * C_k**2 - self.a * D_k + 1
                        C_k = C_k ** 3 - self.a * C_k + b_next
                    b_previous = b_next
                    b_next = b_previous - (C_k - t_m) / D_k
                points.append(b_next)
        points = list(filter(lambda x: abs(x.real) < 3 and abs(x.imag) < 3, points))
        points = [complex_to_pixel(point, res_x=res_x, res_y=res_y, x_range=x_range ,y_range=y_range) for point in points]
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
        ray = self._calculate_ray_mandel(res_x=res_x,
                                         res_y=res_y,
                                         x_range=x_range,
                                         y_range=y_range,
                                         **kwargs)
        d.line(ray, fill=(0, 0, 0),
               width=line_weight, joint="curve")
        return im
    
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

    @staticmethod
    @jit(nopython=True)
    def _f(z, a, b):
        return z**3 - a*z + b
        
    @staticmethod
    @jit(nopython=True)
    def _bottcher(f, z, a, b, max_n=5):
        if a==0 and b==0:
            return z
        else:
            total = 1
            for n in range(1, max_n + 1):
                f_n = z
                for i in range(n-1):
                    f_n = f(f_n, a, b)
                total *= (1 - a/(f_n**2) + b/(f_n**3))**(1/(3**n))
            total = z * total
            return total

    @staticmethod
    @jit(nopython=True)
    def _potential(f, bottcher, z, a, b, max_n=5):
        return math.log(abs(bottcher(f, z, a, b, max_n)))
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_eqpot(f, bottcher, potential, a, b, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), max_n=5):
        results = np.zeros((res_x, res_y))
        step_x = abs((x_range[1] - x_range[0])/res_x)
        step_y = abs((y_range[1] - y_range[0])/res_y)
        for x_i, x in enumerate(np.linspace(x_range[0], x_range[1], res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0], y_range[1], res_y)):
                c1 = complex(x, y)
                c2 = complex(x + step_x, y)
                c3 = complex(x, y + step_y)
                c4 = complex(x + step_x, y + step_y)
                pot1 = potential(f, bottcher, c1, a, b, max_n)
                pot2 = potential(f, bottcher, c2, a, b, max_n)
                pot3 = potential(f, bottcher, c3, a, b, max_n)
                pot4 = potential(f, bottcher, c4, a, b, max_n)
                if min(pot1, pot2, pot3, pot4) <= equipotential <= max(pot1, pot2, pot3, pot4) :
                    results[x_i, y_i] = 1
        return results

    def draw_eqpot(self,
                   im: Image = None,
                   res_x=600,
                   res_y=600,
                   x_range=(-3, 3),
                   y_range=(-3, 3),
                   max_n=5,
                   potential: float = 1.) -> Image.Image:
        """Docstring."""
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
                                      self.a, self.b,
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
        return z - self.cubic(z)/self.cubic.derivative(z)

    def _calculate_mandelbrot(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    @jit(nopython=True)
    def _conv_time_julia(z, a, b, roots, iterations, tol):
        result = [0, 0, 0]
        for i in range(iterations):
            z = 2*z/3 if a == b == 0 else z - (z**3 - a*z + b)/(3*z**2 - a)
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
        for i, r in enumerate(self.cubic.roots):
            for r_ in self.cubic.roots[i+1:]:
                tol = min(tol, abs(r-r_)/3)
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._conv_time_julia,
                                           a=self.cubic.a,
                                           b=self.cubic.b,
                                           roots=self.cubic.roots,
                                           iterations=iterations,
                                           tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        else:
            result_list = map(partial(self._conv_time_julia,
                                      a=self.cubic.a,
                                      b=self.cubic.b,
                                      roots=self.cubic.roots,
                                      iterations=iterations,
                                      tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        return results

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   iterations: int = 32,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   tol: float = 1e-12,
                   multiprocessing: bool = False) -> Image.Image:
        """
        Draw the Julia set for this map with the current parameter values.

        Parameters
        ----------
        res_x: int
            The horizontal resolution of the image.
        res_y: int
            The vertical resolution of the image.
        iterations: int
            The maximum number of times to apply the map iteratively.
        x_range: (float, float)
            The range of x values to consider.
        y_range: (float, float)
            The range of y values to consider.
        tol: float
            The minimum distance before considering the orbit to have
            converged.
        multiprocessing: bool
            Determines whether to use multiprocessing.


        Returns
        -------
        im: Image.Image
            The image of the Mandelbrot set as a Pillow image object.
        """
        results = self._calculate_julia(res_x,
                                        res_y,
                                        iterations,
                                        x_range,
                                        y_range,
                                        tol,
                                        multiprocessing)
        im = Image.fromarray(results[::-1], 'RGB')
        return im

    @ staticmethod
    @ jit(nopython=True)
    def _phi_inv(w_list, roots, a, phi_iters, newt_iters):
        def _q(z, r):
            if r == 0:
                return 1 + 3/(2*z**2)
            return ((9*r**2*z**2 + 18*r**2*z + 9*r**2 - 3*a)
                    / (9*r**2*z**2 + (6*r**2 - 2*a)*z))

        def _dq(z, r):
            if r == 0:
                return -3/z**3
            return -6*(((18*r**4+3*a*r**2)*z**2
                        + (27*r**4-9*a*r**2)*z
                        + 9*r**4-6*a*r**2+a**2)
                       / ((9*r**2*z + 6*r**2 - 2*a)**2*z**2))

        def _f(z, r):
            if r == 0:
                return z**3 + 3/2*z
            return ((9*r**2*z**3 + 18*r**2*z**2 + (9*r**2 - 3*a)*z)
                    / (9*r**2*z + 6*r**2 - 2*a))

        def _df(z, r):
            if r == 0:
                return 3*z**2 + 3/2
            return (6*(9*r**2*z**2 + 9*r**2*z + 3*r**2 - a)
                    * (3*r**2*z + 3*r**2 - a)
                    / (9*r**2*z + 6*r**2 - 2*a)**2)

        def _psi_inv(z, r):
            if r == 0:
                return cmath.sqrt(-a/2)/z
            return (3*r**2*z+3*r**2-a)/(3*r*z)

        z_list = np.zeros((w_list.shape[0]*len(roots), w_list.shape[1]),
                          dtype=np.cdouble)
        for root_idx, r in enumerate(roots):
            pow = 3. if r == 0 else 2.
            for m, row in enumerate(w_list):
                z = row[0]
                for n, w in enumerate(row):
                    for j in range(newt_iters):
                        phi = z * _q(z, r)**(1/pow)
                        dphi = 1/z + _dq(z, r)/(pow*_q(z, r))
                        prev_f = z
                        prev_df = complex(1)
                        for k in range(2, phi_iters):
                            prev_df *= _df(prev_f, r)
                            prev_f = _f(prev_f, r)
                            factor = _q(prev_f, r)**(pow**-k)
                            summand = ((pow**-k)*_dq(prev_f, r)
                                       / _q(prev_f, r)*prev_df)
                            if not (cmath.isnan(factor) or
                                    cmath.isnan(summand)):
                                phi *= factor
                                dphi += summand
                            elif not cmath.isnan(factor):
                                phi *= factor
                            elif not cmath.isnan(summand):
                                dphi += summand
                            else:
                                break
                        if abs(1/dphi*(1-w/phi)) < 1e-8:
                            break
                        z = z-1/dphi*(1-w/phi)
                    z_list[len(roots)*m+root_idx, n] = _psi_inv(z, r)
        return z_list

    def _calculate_ray(self,
                       res_x: int = 600,
                       res_y: int = 600,
                       x_range: tuple = (-3, 3),
                       y_range: tuple = (-3, 3),
                       angles: list = [0.],
                       res_ray: int = 1024,
                       phi_iters: int = 128,
                       newt_iters: int = 256):
        w_list = np.array([[cmath.rect(1/np.sin(r), angle) for r in
                          np.linspace(0, np.pi/2, res_ray+2)[1:-1]]
                          for angle in angles])
        result_list = self._phi_inv(w_list,
                                    self.cubic.roots,
                                    self.cubic.a,
                                    phi_iters,
                                    newt_iters)
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
                 phi_iters: int = 128,
                 newt_iters: int = 128,
                 line_weight: int = 1):
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
        phi_iters: int
            The number of iterations used to approximate phi.
        newt_iters: int
            The number of Newton iterates used to solve the inverse.
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
                                   res_ray=res_ray,
                                   phi_iters=phi_iters,
                                   newt_iters=newt_iters)
        for ray in rays:
            d.line(ray, fill=(255, 255, 255),
                   width=line_weight, joint="curve")
        return im

    @staticmethod
    @jit(nopython=True)
    def _eqpot_points(sample_list,
                      x_space,
                      y_space,
                      roots,
                      a,
                      potentials,
                      phi_iters):
        def _q(z, r):
            if r == 0:
                return 1 + 3/(2*z**2)
            return ((9*r**2*z**2 + 18*r**2*z + 9*r**2 - 3*a)
                    / (9*r**2*z**2 + (6*r**2 - 2*a)*z))

        def _f(z, r):
            if r == 0:
                return z**3 + 3/2*z
            return ((9*r**2*z**3 + 18*r**2*z**2 + (9*r**2 - 3*a)*z)
                    / (9*r**2*z + 6*r**2 - 2*a))

        def psi(z, r):
            if r == 0:
                return cmath.sqrt(-a/2)/z
            return (3*r**2 - a)/(3*r*z - 3*r**2)

        z_list = [False for i in enumerate(sample_list)]
        pots = [0., 0., 0., 0.]
        for root_idx, r in enumerate(roots):
            pow = 3. if r == 0 else 2.
            for j, z in enumerate(sample_list):
                z = psi(z, r)
                phi = z * _q(z, r)**(1/pow)
                prev_f = z
                for k in range(2, phi_iters):
                    prev_f = _f(prev_f, r)
                    factor = _q(prev_f, r)**(pow**-k)
                    if cmath.isnan(factor) or abs(factor) <= 1e-8:
                        break
                    phi *= factor
                for pot in potentials:
                    if abs(np.log(abs(phi)) - pot) <= 8*min(x_space, y_space):
                        z_list[j] = True
                        break
        return z_list

    def _calculate_eqpot(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         potentials: list = [1.],
                         res_eqpot: int = 2048,
                         phi_iters: int = 128,
                         newt_iters: int = 128):
        w_list = np.array([[cmath.rect(np.exp(potential), angle) for angle in
                           np.linspace(-np.pi, np.pi, res_eqpot+1)]
                          for potential in potentials])
        result_list = self._phi_inv(w_list,
                                    self.cubic.roots,
                                    self.cubic.a,
                                    phi_iters,
                                    newt_iters)
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
                   potentials: float = [1.],
                   res_eqpot: int = 2048,
                   phi_iters: int = 128,
                   newt_iters: int = 256,
                   line_weight: int = 1):
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
            The potentials of the lines to plot.
        res_eqpot: float
            The resolution of the equipotential line.
        phi_iters: int
            The number of iterations used to approximate phi.
        newt_iters: int
            The number of Newton iterates used to solve the inverse.
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
                                       potentials=potentials,
                                       res_eqpot=res_eqpot,
                                       phi_iters=phi_iters,
                                       newt_iters=newt_iters)
        for eqpot in eqpots:
            d.point(eqpot, fill=(255, 255, 255))
        return im
