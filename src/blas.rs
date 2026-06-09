use cblas::{Layout, Transpose};
use ndarray::{ArrayRef1, ArrayRef2};



pub fn sgemv(
    transpose: bool, alpha: f32,
    mat: &ArrayRef2<f32>, vec: &ArrayRef1<f32>, out: &mut ArrayRef1<f32>
) {
    let rows = if transpose { vec.len() } else { out.len() };
    let cols = if transpose { out.len() } else { vec.len() };
    let t = if transpose { Transpose::Ordinary } else { Transpose::None };

    assert!(mat.dim() == (rows, cols), "sgemv matrix wrong size");

    let mat = mat.as_slice().expect("sgemv matrix not c-contiguous");
    let vec = vec.as_slice().expect("sgemv input vector not c-contiguous");
    let out = out.as_slice_mut().expect("sgemv output vector not c-contiguous");

    unsafe {
        cblas::sgemv(
            Layout::RowMajor, t, rows as i32, cols as i32, alpha,
            mat, cols as i32, vec, 1, 0., out, 1
        );
    }
}

pub fn sger(
    alpha: f32,
    x: &ArrayRef1<f32>, y: &ArrayRef1<f32>, mat: &mut ArrayRef2<f32>
) {
    assert!(mat.dim() == (x.len(), y.len()), "sger matrix wrong size");

    let mat = mat.as_slice_mut().expect("sger matrix not c-contiguous");
    let x = x.as_slice().expect("sger vector x not c-contiguous");
    let y = y.as_slice().expect("sger vector y not c-contiguous");

    unsafe {
        cblas::sger(
            Layout::RowMajor, x.len() as i32, y.len() as i32, alpha,
            x, 1, y, 1, mat, y.len() as i32
        );
    }
}

pub fn sdot(x: &ArrayRef1<f32>, y: &ArrayRef1<f32>) -> f32 {
    assert!(x.len() == y.len(), "sdot vectors must have same size");

    let x = x.as_slice().expect("sdot vector x not c-contiguous");
    let y = y.as_slice().expect("sdot vector y not c-contiguous");

    unsafe { cblas::sdot(x.len() as i32, x, 1, y, 1) }
}

