mod video_reader;

use video_reader::VideoProbe;
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use cblas::{Layout, Transpose};



#[derive(Parser)]
struct Args {
    video: PathBuf,
    calibration_start: usize,
    calibration_end: usize,
    output: Option<PathBuf>
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let probe = VideoProbe::new(args.video)?;
    let frame_size = probe.width*probe.height;
    let size_approx = frame_size*(probe.n_frames_approx+10);

    let mut frames = vec![0u8; size_approx];
    let mut frame_count = 0;

    println!("Allocated {} for video data", format_bytes(size_approx));
    println!("Reading video...");

    let t = Instant::now();

    for frame in probe.open_reader()? {
        let start = frame_size*frame_count;
        let dst = &mut frames[start..start+frame_size];

        dst.copy_from_slice(&frame?);

        frame_count += 1;
    }

    frames.truncate(frame_count*frame_size);

    println!("Read {frame_count} frames in {} s", t.elapsed().as_secs_f32());

    let calib_start = args.calibration_start;
    let calib_end = args.calibration_end;
    let calib = &frames[calib_start*frame_size..calib_end*frame_size];
    let mut data_mat: Vec<f32> = calib.iter().map(|&x| x as f32).collect();
    let pc_vecs = pca(&mut data_mat, frame_size, 3);

    Ok(())
}

fn format_bytes(n: usize) -> String {
    let unit_idx = ((n as f64).log2()/10.).floor() as usize;
    let unit = ["B", "KiB", "MiB", "GiB", "TiB"][unit_idx];
    let scale = (unit_idx as f64*10.).exp2();

    if unit == "B" { format!("{n} B") }
    else { format!("{:.2} {unit}", n as f64/scale) }
}

fn sgemv(trans: bool, alpha: f32, mat: &[f32], vec: &[f32], out: &mut [f32]) {
    let rows = if trans { vec.len() } else { out.len() };
    let cols = if trans { out.len() } else { vec.len() };
    let t = if trans { Transpose::Ordinary } else { Transpose::None };

    unsafe {
        cblas::sgemv(
            Layout::RowMajor, t, rows as i32, cols as i32, alpha,
            mat, cols as i32, vec, 1, 0., out, 1
        );
    }
}

fn sger(alpha: f32, x: &[f32], y: &[f32], mat: &mut [f32]) {
    unsafe {
        cblas::sger(
            Layout::RowMajor, x.len() as i32, y.len() as i32, alpha,
            x, 1, y, 1, mat, y.len() as i32
        );
    }
}

fn pca(data: &mut [f32], cols: usize, n_pcs: usize) -> Vec<f32> {
    let rows = data.len()/cols;
    let mut pcs_found = 0;
    let mut pcs = vec![0f32; n_pcs*cols];
    let mut r = vec![0f32; cols];
    let mut s = vec![1f32; rows];
    let mut prev_eigval = 0.;

    sgemv(true, 1./rows as f32, data, &s, &mut r);
    sger(-1., &s, &r, data);

    r.fill(1.);

    loop {
        if pcs_found == n_pcs { return pcs; }

        normalise(&mut r);
        sgemv(false, 1., data, &r, &mut s);
        sgemv(true, 1., data, &s, &mut r);

        let eigval = unsafe { cblas::sdot(rows as i32, &s, 1, &s, 1) };

        if prev_eigval > 0. && (prev_eigval-eigval).abs()/eigval < 0.0001 {
            normalise(&mut r);
            sger(-1., &s, &r, data);

            pcs[cols*pcs_found..cols*(pcs_found+1)].copy_from_slice(&r);
            r.fill(1.);

            pcs_found += 1;
            prev_eigval = 0.;
        }
        
        prev_eigval = eigval;
    }
}

fn normalise(vec: &mut [f32]) {
    let norm = vec.iter().map(|&x| (x as f32).powi(2)).sum::<f32>().sqrt();

    for v in vec { *v /= norm; }
}
