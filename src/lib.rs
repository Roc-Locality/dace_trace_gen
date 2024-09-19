use dace_tests::polybench_simplify;
use dace_toolbox::hist::Hist;
use ri::tracing_ri_with_trace;
use std::fs::File;
use std::io::Write;

/// Traces a Polybench benchmark and generates a histogram of memory accesses.
///
/// # Arguments
///
/// * `bench` - The name of the benchmark to trace.
/// * `data_type_size` - The size of the data type used in the benchmark.
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
/// use dace_trace_gen::trace_polybench;
/// trace_polybench("lu", 32, 8, &[]);
/// trace_polybench("syrk", 32, 8, &[16]);
/// ```
pub fn trace_polybench(
    bench: &str,
    data_type_size: usize,
    cache_line_size: usize,
    params: &[usize],
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
        "mvt" => polybench_simplify::mvt(params[0]),
        "trisolv" => polybench_simplify::trisolv(params[0]),
        "syrk" => {
            check_params(2, params);
            polybench_simplify::syrk(params[0], params[1])
        }
        "lu" => polybench_simplify::lu(params[0]),
        "trmm_trace" => {
            check_params(2, params);
            polybench_simplify::trmm_trace(params[0], params[1])
        }
        "lu_affine" => polybench_simplify::lu_affine(params[0]),
        "syr2d" => {
            check_params(2, params);
            polybench_simplify::syr2d(params[0], params[1])
        }
        "gemm" => polybench_simplify::gemm(params[0]),
        "cholesky" => polybench_simplify::cholesky(params[0]),
        "gramschmidt_trace" => {
            check_params(2, params);
            polybench_simplify::gramschmidt_trace(params[0], params[1])
        }
        "3mm" => {
            check_params(4, params);
            polybench_simplify::_3mm(params[0], params[1], params[2], params[3], params[4])
        }
        "2mm" => {
            check_params(3, params);
            polybench_simplify::_2mm(params[0], params[1], params[2], params[3])
        }
        "heat_3d" => {
            check_params(2, params);
            polybench_simplify::heat_3d(params[0], params[1])
        }
        "convolution_2d" => {
            check_params(2, params);
            polybench_simplify::convolution_2d(params[0], params[1])
        }
        "symm" => {
            check_params(2, params);
            polybench_simplify::symm(params[0], params[1])
        }
        _ => panic!("Unknown benchmark"),
    };

    tri.print_structure(0);
    // assign_ref_id(&tri);
    let _hist = tracing_ri_with_trace(&mut tri, data_type_size, cache_line_size);
    // write_hist_to_file(&_hist, "output.csv");
}

#[allow(dead_code)]
fn write_hist_to_file(hist: &Hist, file_path: &str) {
    let mut file = File::create(file_path).expect("Unable to create file");
    write!(file, "{}", hist).expect("Unable to write to file");
}
