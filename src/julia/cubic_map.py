"""Module containing map classes for cubic functions."""
import cmath
from functools import partial
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from PIL import Image, ImageDraw
from time import time

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

        if b == 0:
            self.roots = np.array([complex(0), cmath.sqrt(a), -cmath.sqrt(a)])
            return None
        elif a == 0:
            gamma = complex((-b)**(1/3))
        else:
            gamma = (-b/2+cmath.sqrt(b**2/4-a**3/27))**(1/3)
        omega = cmath.rect(1, 2*np.pi/3)
        omega_ = cmath.rect(1, -2*np.pi/3)
        self.roots = [gamma + a/(3*gamma)]
        if np.all(~np.isclose(gamma*omega + a/(3*gamma)*omega_, self.roots)):
            self.roots.append(gamma*omega + a/(3*gamma)*omega_)
        if np.all(~np.isclose(gamma*omega_ + a/(3*gamma)*omega, self.roots)):
            self.roots.append(gamma*omega_ + a/(3*gamma)*omega)
        self.roots = np.array(self.roots)

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**3 - self.a*z + self.b

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 3*z**2 - self.a

    @ staticmethod
    @ jit(nopython=True)
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

    @ staticmethod
    @ jit(nopython=True)
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

    def external_ray(self, theta, D=20, S=10, R=200, error=0.001):
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

        for i in range(1, D+1):
            for q in range(1, S+1):

                r_m = R ** (1 / (3 ** (i - 1 + q / S)))
                t_m = r_m**(3**(i)) * cmath.exp(2
                                                * np.pi
                                                * 1j
                                                * theta
                                                * 3**(i))
                b_next = points[-1]
                b_previous = 0

                while abs(b_previous - b_next) >= error:
                    C_k = b_next
                    D_k = [0, -self.a + 1]
                    for x in range(i):
                        D_k.append(3 * D_k[-1] * C_k ** 2
                                   - self.a * D_k[-2] + 1)
                        C_k = C_k ** 3 - self.a * C_k + b_next
                    b_previous = b_next
                    b_next = b_previous - (C_k - t_m) / D_k[-1]

                points.append(b_next)

        # filter to be in range [-2,2]
        points = filter(lambda x: abs(x.real) < 2 and abs(x.imag) < 2, points)

        return points

    def draw_ray(self, theta, D=20, S=10, R=50, error=0.1):
        """
        Draw an external ray on matplotlib.

        Oskar - can you add some more description to this docstring, as well as
        types for the parameters.
        """
        results = self.external_ray(theta, D, S, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()


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

    @ staticmethod
    @ jit(nopython=True)
    def _conv_time_julia(z, a, b, roots, iterations, tol):
        result = [0, 0, 0]
        for i in range(iterations):
            z -= (z**3 - a*z + b)/(3*z**2 - a)
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
    @jit(nopython=True)
    def _phi_newton(w_list, roots, a, phi_iters, newt_iters):
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

        z_list = np.zeros((w_list.shape[0]*3, w_list.shape[1]),
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
                    z_list[3*m + root_idx, n] = _psi_inv(z, r)
        return z_list

    def _calculate_ray(self,
                       res_x: int = 600,
                       res_y: int = 600,
                       x_range: tuple = (-3, 3),
                       y_range: tuple = (-3, 3),
                       angles: list = [0.],
                       multiples: int = 1,
                       res_ray: int = 1024,
                       phi_iters: int = 128,
                       newt_iters: int = 256):
        w_list = np.array([[cmath.rect(1/np.sin(r), angle) for r in
                          np.linspace(0, np.pi/2, res_ray+2)[1:-1]]
                          for angle in angles])
        result_list = self._phi_newton(w_list,
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

    def _calculate_eqpot(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         potentials: list = [1.],
                         res_eqpot: int = 1024,
                         phi_iters: int = 128,
                         newt_iters: int = 128):
        w_list = np.array([[cmath.rect(np.exp(potential), angle) for angle in
                           np.linspace(-np.pi, np.pi, res_eqpot+1)[:-1]]
                          for potential in potentials])
        result_list = self._phi_newton(w_list,
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
                                       phi_iters=phi_iters,
                                       newt_iters=newt_iters)
        for eqpot in eqpots:
            d.line(eqpot, fill=(255, 255, 255),
                   width=line_weight, joint="curve")
        return im
