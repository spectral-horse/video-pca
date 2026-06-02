use cblas::{Layout, Transpose};



pub fn sgemv(
    transpose: bool, alpha: f32, mat: &[f32], vec: &[f32], out: &mut [f32]
) {
    assert!(mat.len() == vec.len()*out.len(), "sgemv matrix wrong size");

    let rows = if transpose { vec.len() } else { out.len() };
    let cols = if transpose { out.len() } else { vec.len() };
    let t = if transpose { Transpose::Ordinary } else { Transpose::None };

    unsafe {
        cblas::sgemv(
            Layout::RowMajor, t, rows as i32, cols as i32, alpha,
            mat, cols as i32, vec, 1, 0., out, 1
        );
    }
}

pub fn sger(alpha: f32, x: &[f32], y: &[f32], mat: &mut [f32]) {
    assert!(mat.len() == x.len()*y.len(), "sger matrix wrong size");

    unsafe {
        cblas::sger(
            Layout::RowMajor, x.len() as i32, y.len() as i32, alpha,
            x, 1, y, 1, mat, y.len() as i32
        );
    }
}

pub fn sdot(x: &[f32], y: &[f32]) -> f32 {
    assert!(x.len() == y.len(), "sdot vectors must have same size");

    unsafe { cblas::sdot(x.len() as i32, x, 1, y, 1) }
}

