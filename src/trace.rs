use std::fs::File;
use std::io::Write;

use dace_tests::polybench_simplify;
use dace_toolbox::hist::Hist;
use ri::tracing_ri_with_trace;

/// Traces a Polybench benchmark and generates a histogram of memory accesses.
///
/// # Arguments
///
/// * `bench` - The name of the benchmark to trace.
/// * `data_size` - The size of the data for the benchmark.
/// * `cache_line_size` - The size of the cache line.
/// * `additional_params` - A slice of additional parameters required by some benchmarks.
///
/// # Panics
///
/// This function will panic if the required number of additional parameters is not provided.
///
/// # Examples
///
/// ```
/// trace_polybench("lu", 32, 8, &[]);
/// trace_polybench("syrk", 32, 8, &[16]);
/// ```
pub fn trace_polybench(
    bench: &str,
    cache_line_size: usize,
    data_size: usize,
    additional_params: &[usize],
) {
    fn check_params(required: usize, params: &[usize]) {
        if params.len() < required {
            panic!(
                "Not enough parameters. Required: {}, Provided: {}",
                required,
                params.len()
            );
        }
    }

    let mut tri = match bench {
        "mvt" => polybench_simplify::mvt(data_size),
        "trisolv" => polybench_simplify::trisolv(data_size),
        "syrk" => {
            check_params(1, additional_params);
            polybench_simplify::syrk(data_size, additional_params[0])
        }
        "lu" => polybench_simplify::lu(data_size),
        "trmm_trace" => {
            check_params(1, additional_params);
            polybench_simplify::trmm_trace(data_size, additional_params[0])
        }
        "lu_affine" => polybench_simplify::lu_affine(data_size),
        "syr2d" => {
            check_params(1, additional_params);
            polybench_simplify::syr2d(data_size, additional_params[0])
        }
        "gemm" => polybench_simplify::gemm(data_size),
        "cholesky" => polybench_simplify::cholesky(data_size),
        "gramschmidt_trace" => {
            check_params(1, additional_params);
            polybench_simplify::gramschmidt_trace(data_size, additional_params[0])
        }
        "3mm" => {
            check_params(4, additional_params);
            polybench_simplify::_3mm(
                data_size,
                additional_params[0],
                additional_params[1],
                additional_params[2],
                additional_params[3],
            )
        }
        "2mm" => {
            check_params(3, additional_params);
            polybench_simplify::_2mm(
                data_size,
                additional_params[0],
                additional_params[1],
                additional_params[2],
            )
        }
        "heat_3d" => {
            check_params(1, additional_params);
            polybench_simplify::heat_3d(data_size, additional_params[0])
        }
        "convolution_2d" => {
            check_params(1, additional_params);
            polybench_simplify::convolution_2d(data_size, additional_params[0])
        }
        "symm" => {
            check_params(1, additional_params);
            polybench_simplify::symm(data_size, additional_params[0])
        }
        _ => panic!("Unknown benchmark"),
    };

    tri.print_structure(0);
    // assign_ref_id(&tri);
    let _hist = tracing_ri_with_trace(&mut tri, data_size, cache_line_size);
    // write_hist_to_file(&_hist, "output.csv");
}

#[allow(dead_code)]
fn write_hist_to_file(hist: &Hist, file_path: &str) {
    let mut file = File::create(file_path).expect("Unable to create file");
    write!(file, "{}", hist).expect("Unable to write to file");
}
