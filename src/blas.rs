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

pub fn sgemm(
    transpose_a: bool, transpose_b: bool, alpha: f32,
    a: &ArrayRef2<f32>, b: &ArrayRef2<f32>, c: &mut ArrayRef2<f32>
) {
    let ta = if transpose_a { Transpose::Ordinary } else { Transpose::None };
    let tb = if transpose_b { Transpose::Ordinary } else { Transpose::None };
    let m = if transpose_a { a.ncols() } else { a.nrows() };
    let n = if transpose_b { b.nrows() } else { b.ncols() };
    let ka = if transpose_a { a.nrows() } else { a.ncols() };
    let kb = if transpose_b { b.ncols() } else { b.nrows() };

    assert!(ka == kb, "sgemm matrices have incompatible sizes");

    let a_cols = a.ncols();
    let b_cols = b.ncols();
    let c_cols = c.ncols();
    let a = a.as_slice().expect("sgemm matrix a not c-contiguous");
    let b = b.as_slice().expect("sgemm matrix b not c-contiguous");
    let c = c.as_slice_mut().expect("sgemm matrix c not c-contiguous");

    unsafe {
        cblas::sgemm(
            Layout::RowMajor, ta, tb, m as i32, n as i32, ka as i32, alpha,
            a, a_cols as i32, b, b_cols as i32, 0., c, c_cols as i32
        );
    }
}

/*pub fn sgeqp3(
    a: &mut ArrayRef2<f32>, jpvt: &mut ArrayRef1<i32>, tau: &mut ArrayRef1<f32>
) {
    let (m, n) = a.dim();

    assert!(jpvt.len() == n, "sgeqp3 jpvt wrong length");
    assert!(tau.len() == m.min(n), "sgeqp3 tau wrong length");

    let a = a.as_slice_mut().expect("sgeqp3 matrix a not c-contiguous");
    let jpvt = jpvt.as_slice_mut().expect("sgeqp3 jpvt not c-contiguous");
    let tau = tau.as_slice_mut().expect("sgeqp3 tau not c-contiguous");

    unsafe {
        lapacke::sgeqp3(
            lapacke::Layout::RowMajor,
            m as i32, n as i32, a, n as i32, jpvt, tau
        );
    }
}*/

pub fn sgeqrf(a: &mut ArrayRef2<f32>, tau: &mut ArrayRef1<f32>) {
    let (m, n) = a.dim();

    assert!(tau.len() == m.min(n), "sgeqrf tau wrong length");

    let a = a.as_slice_mut().expect("sgeqrf matrix a not c-contiguous");
    let tau = tau.as_slice_mut().expect("sgeqrf tau not c-contiguous");

    unsafe {
        lapacke::sgeqrf(
            lapacke::Layout::RowMajor, m as i32, n as i32, a, n as i32, tau
        );
    }
}

pub fn sorgqr(a: &mut ArrayRef2<f32>, tau: &ArrayRef1<f32>) {
    let (m, n) = a.dim();
    let k = tau.len();

    assert!(a.dim() == (m, k), "sormqr matrix a wrong size");

    let a = a.as_slice_mut().expect("sorgqr matrix a not c-contiguous");
    let tau = tau.as_slice().expect("sorgqr tau not c-contiguous");

    unsafe {
        lapacke::sorgqr(
            lapacke::Layout::RowMajor, m as i32, n as i32, k as i32,
            a, n as i32, tau
        );
    }
}

pub fn sgesdd(
    a: &mut ArrayRef2<f32>, s: &mut ArrayRef1<f32>,
    u: &mut ArrayRef2<f32>, vt: &mut ArrayRef2<f32>
) {
    let (m, n) = a.dim();
    let n_vecs = m.min(n);

    assert!(s.len() == n_vecs, "sgesdd s array wrong length");
    assert!(u.dim() == (m, n_vecs), "sgesdd u matrix wrong size");
    assert!(vt.dim() == (n_vecs, n), "sgesdd vt matrix wrong size");

    let a = a.as_slice_mut().expect("sgesdd a array not c-contiguous");
    let s = s.as_slice_mut().expect("sgesdd s array not c-contiguous");
    let u = u.as_slice_mut().expect("sgesdd u matrix not c-contiguous");
    let vt = vt.as_slice_mut().expect("sgesdd vt matrix not c-contiguous");

    unsafe {
        lapacke::sgesdd(
            lapacke::Layout::RowMajor, b'S', m as i32, n as i32, a, n as i32,
            s, u, n_vecs as i32, vt, n as i32
        );
    }
}
