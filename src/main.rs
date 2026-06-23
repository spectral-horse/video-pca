mod video_reader;
mod blas;
mod rand;

use blas::*;
use video_reader::VideoProbe;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::{self, Write};
use clap::Parser;
use image::ImageReader;
use anyhow::bail;
use ndarray::prelude::*;



#[derive(Parser)]
struct Args {
    video: PathBuf,
    calibration_start: f32,
    calibration_end: f32,
    output: Option<PathBuf>,

    #[arg(short, long, default_value_t = 3)]
    num_components: usize,

    #[arg(short = 's', long, default_value_t = 10)]
    oversampling: usize,

    #[arg(short, long)]
    mask: Option<PathBuf>
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let probe = VideoProbe::new(args.video)?;

    let mask = match args.mask {
        Some(path) => {
            let img = ImageReader::open(path)?.decode()?.into_luma8();
            let width = img.width() as usize;
            let height = img.height() as usize;

            if width != probe.width || height != probe.height {
                bail!("Mask size does not match video size");
            }

            let values: Vec<bool> = img.pixels().map(|p| p[0] > 0).collect();

            Array2::from_shape_vec((height, width), values).unwrap()
        }
        None => Array2::from_elem((probe.height, probe.width), true)
    };

    let fps = probe.fps;
    let frames = read_video_masked(probe, &mask)?;

    let t = Instant::now();
    let calib_start = (args.calibration_start*fps) as usize;
    let calib_end = (args.calibration_end*fps) as usize;
    let calib = frames.slice(s![calib_start..calib_end, ..]);
    let mut data_mat = calib.mapv(|x| x as f32);

    println!("Allocated {} for calibration data", format_bytes(4*calib.len()));

    let n_pcs = args.num_components;
    let (pc_vecs, variances) = pca(&mut data_mat, n_pcs, 10);
    let variance_max = variances.fold(0f32, |acc, &v| acc.max(v));

    drop(data_mat);

    println!("Computed PCs in {} s", t.elapsed().as_secs_f32());
    println!("Variances:");

    for (i, variance) in variances.indexed_iter() {
        let x = (140.*variance/variance_max) as usize;
        let c = ["", "-"][x%2];
        let width = x/2;

        println!("{i:>2} | {:#<width$}{}", "", c);
    }

    let t = Instant::now();
    let mut buf = Array1::zeros(calib.ncols());
    let mut pc_coords = Array2::zeros((frames.nrows(), n_pcs));

    azip!((frame in frames.rows(), mut pcs in pc_coords.rows_mut()) {
        azip!((b in &mut buf, &pixel in frame) *b = pixel as f32);

        sgemv(false, 1., &pc_vecs, &buf, &mut pcs);
    });

    println!("Transformed data in {} s", t.elapsed().as_secs_f32());

    remove_mean(&mut pc_coords);

    match args.output {
        Some(path) => save_pc_data(
            File::create(path)?, calib_start, calib_end, &pc_coords, fps
        )?,
        None => save_pc_data(
            io::stdout().lock(), calib_start, calib_end, &pc_coords, fps
        )?
    }

    Ok(())
}

fn save_pc_data(
    mut f: impl Write, calib_start: usize, calib_end: usize,
    pc_coords: &ArrayRef2<f32>, fps: f32
) -> io::Result<()> {
    writeln!(f, "# calibration start index = {calib_start}")?;
    writeln!(f, "# calibration end index = {calib_end}")?;

    for (i, row) in pc_coords.rows().into_iter().enumerate() {
        write!(f, "{:e}", i as f32/fps)?;

        for coord in row {
            write!(f, " {coord:e}")?;
        }

        writeln!(f)?;
    }

    Ok(())
}

fn read_video_masked(probe: VideoProbe, mask: &ArrayRef2<bool>)
-> anyhow::Result<Array2<u8>> {
    let frame_size = mask.iter().filter(|&&x| x).count();
    let size_approx = frame_size*(probe.n_frames_approx+10);
    let mut pixels = Vec::with_capacity(size_approx);
    let mut frame_count = 0;

    println!("Allocated {} for video data", format_bytes(size_approx));
    println!("Reading video...");

    let t = Instant::now();

    for frame in probe.open_reader()? {
        let frame = Array2::from_shape_vec(mask.dim(), frame?).unwrap();

        azip!((&m in mask, &src in &frame) {
            if m { pixels.push(src); }
        });

        frame_count += 1;
    }

    println!("Read {frame_count} frames in {} s", t.elapsed().as_secs_f32());

    let shape = (frame_count, frame_size);
    let frames = Array2::from_shape_vec(shape, pixels).unwrap();

    Ok(frames)
}

fn format_bytes(n: usize) -> String {
    let unit_idx = ((n as f64).log2()/10.).floor() as usize;
    let unit = ["B", "KiB", "MiB", "GiB", "TiB"][unit_idx];
    let scale = (unit_idx as f64*10.).exp2();

    if unit == "B" { format!("{n} B") }
    else { format!("{:.2} {unit}", n as f64/scale) }
}

fn remove_mean(data: &mut ArrayRef2<f32>) {
    let ones = Array1::from_elem(data.nrows(), 1.);
    let mut mean = Array1::zeros(data.ncols());

    sgemv(true, 1., data, &ones, &mut mean);
    sger(-1./data.nrows() as f32, &ones, &mean, data);
}

fn pca(data: &mut ArrayRef2<f32>, n_pcs: usize, oversampling: usize)
-> (Array2<f32>, Array1<f32>) {
    remove_mean(data);

    let (m, n) = data.dim();
    let l = n_pcs+oversampling;
    let omega = Array2::from_shape_fn((n, l), |_| rand::rand_normal());
    let mut tau = Array1::zeros(l);
    let mut y = Array2::zeros((m, l));
    let mut b = Array2::zeros((l, n));
    
    sgemm(false, false, 1., &data, &omega, &mut y);
    sgeqrf(&mut y, &mut tau);
    sorgqr(&mut y, &tau);
    sgemm(true, false, 1., &y, &data, &mut b);

    let mut s = Array1::zeros(l);
    let mut u = Array2::zeros((l, l));
    let mut vt = Array2::zeros((l, n));

    sgesdd(&mut b, &mut s, &mut u, &mut vt);

    let pc_vecs = vt.slice(s![..n_pcs, ..]).into_owned();
    let variances = s.slice(s![..n_pcs]).mapv(|v| v*v);

    (pc_vecs, variances)
}
