mod video_reader;

use video_reader::VideoProbe;
use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;



#[derive(Parser)]
struct Args {
    video: PathBuf,
    calibration_start: usize,
    calibration_end: usize
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let probe = VideoProbe::new(args.video)?;
    let frame_size = probe.width*probe.height;
    let size_approx = frame_size*(probe.n_frames_approx+10);

    println!("Allocating {} for video data", format_bytes(size_approx));

    let mut frames = Vec::<u8>::with_capacity(size_approx);

    unsafe { frames.set_len(size_approx); }

    println!("Reading video...");

    let t = Instant::now();

    for (frame_pos, frame) in probe.open_reader()?.enumerate() {
        let start = frame_size*frame_pos;
        let dst = &mut frames[start..start+frame_size];

        dst.copy_from_slice(&frame?);
    }

    println!("Read in {} s", t.elapsed().as_secs_f32());

    Ok(())
}

fn format_bytes(n: usize) -> String {
    let unit_idx = ((n as f64).log2()/10.).floor() as usize;
    let unit = ["B", "KiB", "MiB", "GiB", "TiB"][unit_idx];
    let scale = (unit_idx as f64*10.).exp2();

    if unit == "B" { format!("{n} B") }
    else { format!("{:.2} {unit}", n as f64/scale) }
}
