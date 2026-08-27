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
/// Extract the principal components of a time-varying image sequence (a video).
/// 
/// This tool performs principal component analysis on an image sequence read
/// from a video file, optionally masking each frame with a boolean mask, and
/// can output the principal component values along with associated variances.
/// This is done by randomised SVD (Halko).
struct Args {
    video: PathBuf,

    #[arg(long, default_value_t = 0.)]
    t0: f32,

    #[arg(long)]
    t1: Option<f32>,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(short, long)]
    variance_output: Option<PathBuf>,

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
    let calib_start = (args.t0*fps) as usize;
    let calib_end = match args.t1 {
        Some(t1) => (t1*fps) as usize,
        None => frames.nrows()
    };

    let calib = frames.slice(s![calib_start..calib_end, ..]);
    let n_pcs = args.num_components;
    let (pc_vecs, variances) = pca(&calib, n_pcs, 10);

    println!("Computed PCs in {} s", t.elapsed().as_secs_f32());
    println!("Variances:");
    print_bar_chart(&variances, 70);

    let t = Instant::now();
    let mut buf = Array1::zeros(calib.ncols());
    let mut pc_coords = Array2::zeros((frames.nrows(), n_pcs));

    azip!((frame in frames.rows(), mut pcs in pc_coords.rows_mut()) {
        azip!((b in &mut buf, &pixel in frame) *b = pixel as f32);

        sgemv(false, 1., &pc_vecs, &buf, &mut pcs);
    });

    println!("Transformed data in {} s", t.elapsed().as_secs_f32());

    pc_coords -= &row_mean(&pc_coords);

    match args.output {
        Some(path) => save_pc_data(
            File::create(path)?, calib_start, calib_end, &pc_coords, fps
        )?,
        None => save_pc_data(
            io::stdout().lock(), calib_start, calib_end, &pc_coords, fps
        )?
    }

    if let Some(path) = args.variance_output {
        let mut f = File::create(path)?;

        writeln!(f, "# PC index, variance, fraction of max variance")?;

        for (i, v) in variances.iter().enumerate() {
            writeln!(f, "{i} {v} {}", v/variances[0])?;
        }
    }

    Ok(())
}

fn print_bar_chart(values: &ArrayRef1<f32>, max_width: usize) {
    let max = values.fold(0f32, |acc, &v| acc.max(v));

    for (i, value) in values.indexed_iter() {
        let x = (max_width as f32*2.*value/max) as usize;
        let c = ["", "-"][x%2];
        let width = x/2;

        println!("{i:>2} | {:#<width$}{}", "", c);
    }
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

fn row_mean<A: Copy + Into<f32>>(data: &ArrayRef2<A>) -> Array1<f32> {
    let n = data.nrows() as f32;

    data.fold_axis(Axis(0), 0f32, |&acc, &v| acc+v.into())
        .mapv_into(|v| v/n)
}

fn pca(data: &ArrayRef2<u8>, n_pcs: usize, oversampling: usize)
-> (Array2<f32>, Array1<f32>) {
    let rmean = row_mean(data);
    let (m, n) = data.dim();
    let l = n_pcs+oversampling;
    let omega = Array2::from_shape_fn((n, l), |_| rand::rand_normal());
    let mut buf = Array1::zeros(n);
    let mut tau = Array1::zeros(l);
    let mut y = Array2::zeros((m, l));
    let mut b = Array2::zeros((l, n));

    azip!((data_row in data.rows(), mut y_row in y.rows_mut()) {
        azip!((v in &mut buf, &d in data_row, &m in &rmean) *v = d as f32-m);

        sgemv(true, 1., &omega, &buf, &mut y_row);
    });

    sgeqrf(&mut y, &mut tau);
    sorgqr(&mut y, &tau);

    azip!((data_row in data.rows(), y_row in y.rows()) {
        azip!((v in &mut buf, &d in data_row, &m in &rmean) *v = d as f32-m);

        sger(1., &y_row, &buf, &mut b);
    });

    let mut s = Array1::zeros(l);
    let mut u = Array2::zeros((l, l));
    let mut vt = Array2::zeros((l, n));

    sgesdd(&mut b, &mut s, &mut u, &mut vt);

    let pc_vecs = vt.slice(s![..n_pcs, ..]).into_owned();
    let variances = s.slice(s![..n_pcs]).mapv(|v| v*v);

    (pc_vecs, variances)
}
