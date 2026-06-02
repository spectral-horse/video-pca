mod video_reader;
mod blas;

use blas::*;
use video_reader::VideoProbe;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use clap::Parser;
use image::ImageReader;
use anyhow::bail;



#[derive(Parser)]
struct Args {
    video: PathBuf,
    calibration_start: usize,
    calibration_end: usize,
    output: Option<PathBuf>,

    #[arg(short, long, default_value_t = 3)]
    num_components: usize,

    #[arg(short, long, default_value_t = 1e-4)]
    precision: f32,

    #[arg(short, long)]
    mask: Option<PathBuf>
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let probe = VideoProbe::new(args.video)?;

    let mask: Vec<bool> = match args.mask {
        Some(path) => {
            let img = ImageReader::open(path)?.decode()?.into_luma8();
            let width = img.width() as usize;
            let height = img.height() as usize;

            if width != probe.width || height != probe.height {
                bail!("Mask size does not match video size");
            }

            img.pixels().map(|p| p[0] > 0).collect()
        }
        None => vec![true; probe.width*probe.height]
    };

    let frame_size = mask.iter().map(|&x| x as usize).sum();
    let size_approx = frame_size*(probe.n_frames_approx+10);

    let mut frames = vec![0u8; size_approx];
    let mut frame_count = 0;
    let mut pixel_pos = 0;

    println!("Allocated {} for video data", format_bytes(size_approx));
    println!("Reading video...");

    let t = Instant::now();

    for frame in probe.open_reader()? {
        for (&m, src) in mask.iter().zip(frame?) {
            if m {
                frames[pixel_pos] = src;
                pixel_pos += 1;
            }
        }

        frame_count += 1;
    }

    frames.truncate(pixel_pos);

    println!("Read {frame_count} frames in {} s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let calib_start = args.calibration_start;
    let calib_end = args.calibration_end;
    let calib = &frames[calib_start*frame_size..calib_end*frame_size];
    let mut data_mat: Vec<f32> = calib.iter().map(|&x| x as f32).collect();

    println!(
        "Allocated {} for calibration data",
        format_bytes(4*frame_size*(calib_end-calib_start))
    );

    let n_pcs = args.num_components;
    let pc_vecs = pca(&mut data_mat, frame_size, n_pcs, args.precision);

    drop(data_mat);

    println!("Computed PCs in {} s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let mut buf = vec![0f32; frame_size];
    let mut pc_coords = vec![0f32; n_pcs*frame_count];

    for (i, frame) in frames.chunks(frame_size).enumerate() {
        for (b, &pixel) in buf.iter_mut().zip(frame) {
            *b = pixel as f32;
        }

        sgemv(false, 1., &pc_vecs, &buf, &mut pc_coords[n_pcs*i..n_pcs*(i+1)]);
    }

    println!("Transformed data in {} s", t.elapsed().as_secs_f32());

    let mut f: Box<dyn Write> = match args.output {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(std::io::stdout().lock())
    };

    for i in 0..frame_count {
        write!(f, "{:e}", i as f32/probe.fps)?;

        for j in 0..n_pcs {
            write!(f, " {:e}", pc_coords[i*n_pcs+j])?;
        }

        writeln!(f)?;
    }

    Ok(())
}

fn format_bytes(n: usize) -> String {
    let unit_idx = ((n as f64).log2()/10.).floor() as usize;
    let unit = ["B", "KiB", "MiB", "GiB", "TiB"][unit_idx];
    let scale = (unit_idx as f64*10.).exp2();

    if unit == "B" { format!("{n} B") }
    else { format!("{:.2} {unit}", n as f64/scale) }
}

fn pca(data: &mut [f32], cols: usize, n_pcs: usize, precision: f32)
-> Vec<f32> {
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

        let eigval = sdot(&s, &s);

        if prev_eigval > 0. && (prev_eigval-eigval).abs()/eigval < precision {
            normalise(&mut r);
            sger(-1., &s, &r, data);

            pcs[cols*pcs_found..cols*(pcs_found+1)].copy_from_slice(&r);
            r.fill(1.);

            pcs_found += 1;
            prev_eigval = 0.;
        }
        else { prev_eigval = eigval; }
    }
}

fn normalise(vec: &mut [f32]) {
    let norm = sdot(vec, vec).sqrt();

    for v in vec { *v /= norm; }
}
