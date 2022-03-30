
use ndarray;
use numpy::{PyArrayDyn, IntoPyArray, PyReadonlyArrayDyn, PyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use num_complex::Complex;
use itertools_num::linspace;

/// A Python module implemented in Rust.
#[pymodule]
fn rustlib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    
    #[pyfn(m)]
    fn factorial(_py: Python<'_>, n: usize) -> PyResult<usize> {
        let mut fac = 1;
        for i in 1..(n + 1) {
            fac *= i;
        };
        Ok(fac)
    }

    #[pyfn(m)]
    fn axb<'py>(py: Python<'py>, a: f64, b: PyReadonlyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let b = b.as_array();
        let res = a * &b;
        res.into_pyarray(py)
    }

    #[pyfn(m)]
    fn calculate_mandelbrot_quadratic<'py>(
        py: Python<'py>, res_x: usize, res_y: usize, iterations: usize,
        x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_max: f64, 
    ) -> &PyArray2<f64> {
        let mut result_array = ndarray::Array2::<f64>::zeros((res_y, res_x));
        let x_list = linspace(x_min, x_max, res_x);

        for (j, x) in x_list.enumerate() {
            let y_list = linspace(y_min, y_max, res_y);
            for (i, y) in y_list.enumerate() {
                let c = Complex::<f64>::new(x, y);
                let mut z = Complex::<f64>::new(0.0, 0.0);
                let mut k = 0;
                while k < iterations && z.norm() < z_max {
                    z = z.powu(2) + c;
                    k += 1;
                }
                result_array[[i, j]] = (k as f64) / (iterations as f64);
            }
        }
        result_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn calculate_julia_quadratic<'py>(
        py: Python<'py>, c: Complex<f64>, res_x: usize, res_y: usize, iterations: usize,
        x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_max: f64,
    ) -> &PyArray2<f64> {
        let mut result_array = ndarray::Array2::<f64>::zeros((res_y, res_x));
        let x_list = linspace(x_min, x_max, res_x);

        for (j, x) in x_list.enumerate() {
            let y_list = linspace(y_min, y_max, res_y);
            for (i, y) in y_list.enumerate() {
                let mut z = Complex::<f64>::new(x, y);
                let mut k = 0;
                while k < iterations && z.norm() < z_max {
                    z = z.powu(2) + c;
                    k += 1;
                }
                result_array[[i, j]] = (k as f64) / (iterations as f64)
            }
        }
        result_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn calculate_mandelbrot_cubic<'py>(
        py: Python<'py>, a: Complex<f64>, res_x: usize, res_y: usize, iterations: usize,
        x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_max: f64,
    ) -> &PyArray2<f64> {
        let mut result_array = ndarray::Array2::<f64>::zeros((res_y, res_x));
        let x_list = linspace(x_min, x_max, res_x);

        for (j, x) in x_list.enumerate() {
            let y_list = linspace(y_min, y_max, res_y);
            for (i, y) in y_list.enumerate() {
                let b = Complex::<f64>::new(x, y);
                let c1 = - (a / 3.0).sqrt();
                let c2 = (a / 3.0).sqrt();
                let mut z1 = c1;
                let mut z2 = c2;
                let mut k = 0;
                while k < iterations && (z1 - c1).norm() < z_max && (z2 - c2).norm() < z_max {
                    z1 = z1.powu(3) - a*z1 + b;
                    z2 = z2.powu(3) - a*z2 + b;
                    k += 1;
                }
                result_array[[i, j]] = (k as f64) / (iterations as f64)
            }
        }
        result_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn calculate_julia_cubic<'py>(
        py: Python<'py>, a: Complex<f64>, b: Complex<f64>, res_x: usize, res_y: usize, iterations: usize,
        x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_max: f64,
    ) -> &PyArray2<f64> {
        let mut result_array = ndarray::Array2::<f64>::zeros((res_y, res_x));
        let x_list = linspace(x_min, x_max, res_x);

        for (j, x) in x_list.enumerate() {
            let y_list = linspace(y_min, y_max, res_y);
            for (i, y) in y_list.enumerate() {
                let mut z = Complex::<f64>::new(x, y);
                let mut k = 0;
                while k < iterations && z.norm() < z_max {
                    z = z.powu(3) - a*z + b;
                    k += 1;
                }
                result_array[[i, j]] = (k as f64) / (iterations as f64)
            }
        }
        result_array.into_pyarray(py)
    }

    Ok(())
}