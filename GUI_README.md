# GUI for M2R-Julia-Sets
The interactive GUI plots the Mandelbrot set for a function alongside the Julia set. You can visualise either the quadratic or the cubic Mandelbrot and Julia sets.

## Usage
First, install the package via the instructions in the [README](./README.md).

Then, to run the GUI, use the command `python -m julia`. The default is to use the quadratic function z^2 + c, to use the cubic function z^3 - az + b use the command `python -m julia -c`, and to use the Newton mapping for either function, use the `-n` option. To enable multiprocessing, use the `-m` option.

## Controls
* **Zooming** - To zoom in on an area of a plot, click and drag the area you'd like to zoom in on.
* **Choosing a** - To choose a new value for a, in the cubic z^3 - az + b, left click any point on the Mandelbrot plot and the corresponding complex number will be the new value.
* **Choosing b/c** - To choose a new value for b in the cubic or c in the quadratic, for the Julia set, right click any point on the plot and this complex number will be the new value.
* **Reset** - Press the `m` key to reset the Mandelbrot set to its original zoom and the `j` key to reset the Julia set.
* **External/Internal rays** - To draw external rays, press the `r` key and use the dialog box to draw the chosen rays.
* **Equipotential lines** - To draw equipotential lines, press the `e` key and use the dialog box to draw the chosen lines.
* **Quit** - Press the `q` key to quit the application.