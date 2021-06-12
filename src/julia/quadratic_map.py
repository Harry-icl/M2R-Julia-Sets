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

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**2 + self.c

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 2*z

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(c, iterations, z_max):
        z = c
        for i in range(iterations):
            z = z**2 + c
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, c, iterations, z_max):
        for i in range(iterations):
            z = z**2 + c
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

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

    def external_ray(self, theta, D=50, S=20, R=200, error=0.1):
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
                    for x in range(i):
                        D_k = 2 * D_k * C_k + 1
                        C_k = C_k ** 2 + c_next
                    c_previous = c_next
                    c_next = c_previous - (C_k - t_m) / D_k
                points.append(c_next)
        points = filter(lambda x: abs(x.real) < 3 and abs(x.imag) < 3, points)
        return points

    def draw_ray(self, theta, D=50, S=20, R=200, error=0.001):

        results = self.external_ray(theta, D, S, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()
    
    def composition(self, z, iters):
        result = z
        for i in range(iters):
            result = self.__call__(result)
        return result

    def newton_map_julia(self, z, n, R, theta, error=0.001):
        new_result = z
        old_result = 0
        C_k = z
        D_k = 1
        for j in range(n):
            D_k = 2*D_k*C_k
            C_k = C_k**2 + self.c
        print(C_k, D_k)
        while abs(old_result - new_result) >= error:
            old_result = new_result
            new_result = old_result - (C_k - R*cmath.exp(2 * np.pi * 1j * theta * 2**n))/D_k 
            #print(new_result)
        return new_result

    def newton_map_julia_log(self, z, n, R, theta, error=0.001):
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


    def external_ray_julia(self, theta, D=50, R=50, error=0.01):
        points = [R * cmath.exp(2 * np.pi * theta * 1j)]
        for i in range(2, D):
                point = points[-1]
                point = self.newton_map_julia(point, i, R, theta, error)
                points.append(point)
                #print(point)
        #points = filter(lambda x: abs(x.real) < 3 and abs(x.imag) < 3, points)
        points
        return points

    '''def external_ray_julia(self, theta, D=50, R=200, error=0.0001):
        results = []
        list_points = [((1/(2**i))*cmath.log(R) + 2 * np.pi * theta * 1j) for i in range(D)]
        for i in range(D):
            result = list_points[i]
            result = self.newton_map_julia_log(result, i, R, theta, error)  # i or i+2
            results.append(result)
        results = filter(lambda x: abs(x.real) < 3 and abs(x.imag) < 3, results)
        return results'''

    def draw_ray_julia(self, theta, D=50, R=50, error=0.001):

        results = self.external_ray_julia(theta, D, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()

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
    def _calculate_equipotential(f, bottcher, potential, c, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), max_n=5):
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
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_equipotential_complex(f, bottcher, potential, c, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), max_n=5):
        results = []
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
                    results.append(c1)
        return results
    
    def draw_equipotential(self, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), max_n=5) -> Image.Image:

        results = self._calculate_equipotential(self._f, self._bottcher, self._potential, self.c, equipotential, res_x, res_y, x_range, y_range, max_n)
        results = np.rot90(results)
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
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
        return z - self.quadratic.derivative(z)/self.quadratic(z)

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(c,
                                iterations,
                                z_max):
        z = c
        for i in range(iterations):
            z = z - (2*z)/(z**2 + c)
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z,
                           c,
                           iterations,
                           z_max):
        for i in range(iterations):
            z = z - (2*z) / (z**2 + c)
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-2, 2),
                              y_range: tuple = (-2, 2),
                              z_max: float = 2,
                              multiprocessing: bool = False):
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
                         multiprocessing: bool = False):
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        pass

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   line_weight: int = 1) -> Image.Image:
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        if self.quadratic.c == 0:
            return im
        d = ImageDraw.Draw(im)
        real = cmath.sqrt(self.quadratic.c).real
        imag = cmath.sqrt(self.quadratic.c).imag
        if abs(real) >= abs(imag):
            d.line([0,
                    round((x_range[0]*imag/real-y_range[1])
                          / (y_range[0]-y_range[1])*(res_y-1)),
                    res_x-1,
                    round((x_range[1]*imag/real-y_range[1])
                          / (y_range[0]-y_range[1])*(res_y-1))],
                   fill=0,
                   width=line_weight)
        else:
            d.line([round((y_range[1]*real/imag-x_range[0])
                          / (x_range[1]-x_range[0])*(res_y-1)),
                    0,
                    round((y_range[0]*real/imag-x_range[0])
                          / (x_range[1]-x_range[0])*(res_y-1)),
                    res_y-1],
                   fill=0,
                   width=line_weight)
        im.show()
        return im

    def _inv_bottcher(self, z: complex):
        return 1j*cmath.sqrt(self.quadratic.c)*(z+1)/(z-1)

    def _complex_to_pixel(self,
                          z: complex,
                          res_x: int = 600,
                          res_y: int = 600,
                          x_range: tuple = (-3, 3),
                          y_range: tuple = (-3, 3)) -> tuple:
        return (round((z.real-x_range[0])/(x_range[1]-x_range[0])*(res_x-1)),
                round((z.imag-y_range[1])/(y_range[0]-y_range[1])*(res_y-1)))

    def _calculate_rays(self,
                        res_x: int = 600,
                        res_y: int = 600,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3),
                        multiples: int = 12,
                        res_ray: int = 1024):
        return [[self._complex_to_pixel(
            self._inv_bottcher(cmath.rect(1/np.cos(r), phi)),
            res_x,
            res_y,
            x_range,
            y_range
        )
            for r in np.linspace(0, np.pi/2, res_ray+2)[1:-1]]
            for phi in np.linspace(0, 2*np.pi, multiples+1)[:-1]]

    def draw_rays(self,
                  res_x: int = 600,
                  res_y: int = 600,
                  x_range: tuple = (-3, 3),
                  y_range: tuple = (-3, 3),
                  multiples: int = 12,
                  res_ray: int = 1024,
                  line_weight: int = 1) -> Image.Image:
        im = self.draw_julia(res_x, res_y, x_range, y_range, line_weight)
        d = ImageDraw.Draw(im)
        for ray in self._calculate_rays(res_x,
                                        res_y,
                                        x_range,
                                        y_range,
                                        multiples,
                                        res_ray):
            d.line(ray, fill=0, width=line_weight, joint="curve")
        im.show()
        return im

    def _calculate_eqpots(self,
                          res_x: int = 600,
                          res_y: int = 600,
                          x_range: tuple = (-3, 3),
                          y_range: tuple = (-3, 3),
                          levels: int = 12,
                          res_eqpot: int = 1024):
        return [[self._complex_to_pixel(
            self._inv_bottcher(cmath.rect(1/np.cos(r), phi)),
            res_x,
            res_y,
            x_range,
            y_range
        )
            for phi in np.linspace(0, 2*np.pi, res_eqpot+1)[:-1]]
            for r in np.linspace(0, np.pi/2, levels+2)[1:-1]]

    def draw_eqpots(self,
                    res_x: int = 600,
                    res_y: int = 600,
                    x_range: tuple = (-3, 3),
                    y_range: tuple = (-3, 3),
                    levels: int = 12,
                    res_eqpot: int = 1024,
                    line_weight: int = 1) -> Image.Image:
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        d = ImageDraw.Draw(im)
        for eqpot in self._calculate_eqpots(res_x,
                                            res_y,
                                            x_range,
                                            y_range,
                                            levels,
                                            res_eqpot):
            d.line(eqpot, fill=0, width=line_weight, joint="curve")
        im.show()
        return im
