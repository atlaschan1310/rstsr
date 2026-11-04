//! Kernel collections (per-family modules).
//!
//! - [`plan`]: host-side layout analysis (flat / magic-divmod / generic).
//! - [`elementwise`]: planned elementwise launches (unary/binary/scalar/ where families) with flat,
//!   parameterized, and rank-generic variants.
//! - [`exprs`]: dtype-dispatch element-expression builders.
//! - `rstsr_complex.cuh`: complex-math device header (spliced into generated sources at the marker;
//!   see dtype.rs complex INCLUDES).

pub(crate) mod elementwise;
pub(crate) mod exprs;
pub(crate) mod ops_binary_extra;
pub(crate) mod ops_ternary;
pub(crate) mod plan;
pub(crate) mod reduce_plan;
pub(crate) mod reduce;

/// Complex-math device header (include_str! keeps it a real .cuh file).
pub(crate) const COMPLEX_HEADER: &str = include_str!("rstsr_complex.cuh");

/// Splice the complex header at its marker in generated source lines.
pub(crate) fn splice_includes(includes: &[&str]) -> String {
    let mut s = String::new();
    let mut seen: Vec<&str> = Vec::new(); // exact-line dedupe
    for inc in includes {
        if inc.contains("@RSTSR_COMPLEX_HEADER@") {
            if seen.iter().any(|x| x.contains("@RSTSR_COMPLEX_HEADER@")) {
                continue;
            }
            seen.push(inc);
            s.push_str(COMPLEX_HEADER);
        } else if !seen.contains(inc) {
            seen.push(inc);
            s.push_str(inc);
        }
        s.push('\n');
    }
    s
}
