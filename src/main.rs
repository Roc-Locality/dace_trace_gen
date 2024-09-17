#![feature(get_mut_unchecked)]

use dace::construct;
use ri::tracing_ri_with_trace;

mod trace;

fn main() {
    // Choose the target based on some configuration or input
    let target_type = "symm"; // This can be changed to "lu", "symm", or "gemver"

    match target_type {
        "lu" => trace::trace_polybench("lu", 8, 32, &[]),
        "symm" => trace::trace_polybench("symm", 8, 32, &[32]),
        // "gemver" => trace::trace_polybench("gemver", 8, 1024, &[]),
        "gram" => trace::trace_polybench("gramschmidt_trace", 8, 32, &[32]),
        "mm" => perform_matrix_operations(32),
        _ => panic!("Unknown target type"),
    };
}

#[allow(dead_code)]
fn perform_matrix_operations(n: i32) {
    // creating C[i,j] += A[i,k] * B[k,j]
    let mut nested_loops = construct::nested_loops(&["i", "j", "k"], n);

    let ref_c = construct::squ_ref("c", n, vec!["i", "j"]);
    let ref_a = construct::squ_ref("a", n, vec!["i", "k"]);
    let ref_b = construct::squ_ref("b", n, vec!["k", "j"]);

    [ref_c, ref_a, ref_b].iter_mut().for_each(|ref_node| {
        construct::insert_at_innermost(ref_node, &mut nested_loops);
    });

    tracing_ri_with_trace(&mut nested_loops, 8, 32);
}
