"""Tools for the interactive GUI."""


def to_complex(x, y, x_range, y_range, x_res, y_res):
    """
    Convert coordinates from window to a complex number.

    Parameters
    ----------
    x: int
        The x coordinate.
    y: int
        The y coordinate.
    x_range: (float, float)
        The range of x values in the window.
    y_range: (float, float)
        The range of y values in the window.
    x_res: float
        The horizontal resolution of the window.
    y_res: float
        The vertical resolution of the window.

    Returns
    -------
    z: complex
        The corresponding complex number.
    """
    x_val = x_range[0] + ((x / x_res)
                          * (x_range[1] - x_range[0]))
    y_val = y_range[1] - ((y / y_res)
                          * (y_range[1] - y_range[0]))
    return complex(x_val, y_val)


def from_complex(z, x_range, y_range, x_res, y_res):
    """
    Convert complex number to window coordinates.

    Parameters
    ----------
    z: complex
        The complex number to convert.
    x_range: (float, float)
        The range of x values in the window.
    y_range: (float, float)
        The range of y values in the window.
    x_res: int
        The horizontal resolution of the window.
    y_res: int
        The vertical resolution of the window.

    Returns
    -------
    x, y: int, int
        The coordinates of the corresponding point.
    """
    x = (z.real - x_range[0]) * x_res / (x_range[1] - x_range[0])
    y = (y_range[1] - z.imag) * y_res / (y_range[1] - y_range[0])
    return int(x), int(y)


def title_generator(a, x_range, y_range):
    """
    Generate a window title for GUI.

    Parameters
    ----------
    a: complex
        The a parameter in the cubic.
    x_range: tuple
        The range of x values displayed.
    y_range: tuple
        The range of y values displayed.

    Returns
    -------
    title: str
        The window title.
    """
    func_name = (f"z^3 - ({round(a.real, 3)} + {round(a.imag, 3)}i)z + b"
                 if a.imag >= 0
                 else
                 f"z^3 - ({round(a.real, 3)} {round(a.imag, 3)}i)z + b")
    bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i"
                   if y_range[0] >= 0
                   else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
    top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i"
                 if y_range[1] >= 0
                 else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
    return f"Mandelbrot set of {func_name}, ({bottom_left}, {top_right})"


def title_generator_julia(a, b, x_range, y_range):
    """
    Generate a window title for GUI for the Julia set.

    Parameters
    ----------
    a: complex
        The a parameter in the cubic.
    b: complex
        The b parameter in the cubic.
    x_range: tuple
        The range of x values displayed.
    y_range: tuple
        The range of y values displayed.

    Returns
    -------
    title: str
        The window title.
    """
    a_repr = (f"({round(a.real, 3)} + {round(a.imag, 3)}i)"
              if a.imag >= 0
              else f"({round(a.real, 3)} {round(a.imag, 3)}i)")
    b_repr = (f"({round(b.real, 3)} + {round(b.imag, 3)}i)"
              if b.imag >= 0
              else f"({round(b.real, 3)} {round(b.imag, 3)}i)")

    func_name = f"z^3 - {a_repr}z + {b_repr}"
    bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i"
                   if y_range[0] >= 0
                   else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
    top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i"
                 if y_range[1] >= 0
                 else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
    return f"Julia set of {func_name}, ({bottom_left}, {top_right})"


def title_generator_quad(x_range, y_range):
    """
    Generate a window title for GUI.

    Parameters
    ----------
    x_range: tuple
        The range of x values displayed.
    y_range: tuple
        The range of y values displayed.

    Returns
    -------
    title: str
        The window title.
    """
    func_name = "z^2 + c"
    bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i"
                   if y_range[0] >= 0
                   else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
    top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i"
                 if y_range[1] >= 0
                 else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
    return f"Mandelbrot set of {func_name}, ({bottom_left}, {top_right})"


def title_generator_quad_julia(c, x_range, y_range):
    """
    Generate a window title for GUI for the Julia set.

    Parameters
    ----------
    c: complex
        The c parameter in the quadratic.
    x_range: tuple
        The range of x values displayed.
    y_range: tuple
        The range of y values displayed.

    Returns
    -------
    title: str
        The window title.
    """
    func_name = (f"z^2 + ({round(c.real, 3)} + {round(c.imag, 3)})i"
                 if c.imag >= 0
                 else f"z^2 + ({round(c.real, 3)} {round(c.imag, 3)})i")
    bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i"
                   if y_range[0] >= 0
                   else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
    top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i"
                 if y_range[1] >= 0
                 else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
    return f"Julia set of {func_name}, ({bottom_left}, {top_right})"
