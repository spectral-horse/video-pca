use std::io::{self, Read};
use std::path::{Path, PathBuf};
use duct::{cmd, ReaderHandle};



pub struct VideoProbe {
    pub path: PathBuf,
    pub n_frames_approx: usize,
    pub width: usize,
    pub height: usize,
    pub fps: f32
}

pub struct VideoReader {
    proc: ReaderHandle,
    frame_size: usize,
    finished: bool
}

impl VideoProbe {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let output = cmd!(
            "ffprobe",
            "-i", path.as_ref(),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration",
            "-output_format", "csv=p=0"
        ).read()?;

        let parts: Vec<&str> = output.trim().split(&[',', '/']).collect();
        let width: usize = parts[0].parse().unwrap();
        let height: usize = parts[1].parse().unwrap();
        let fps_numerator: f32 = parts[2].parse().unwrap();
        let fps_denominator: f32 = parts[3].parse().unwrap();
        let duration: f32 = parts[4].parse().unwrap();
        let fps = fps_numerator/fps_denominator;
        let n_frames_approx = (fps*duration).round() as usize;

        Ok(Self {
            path: path.as_ref().to_path_buf(),
            n_frames_approx, width, height, fps
        })
    }

    pub fn open_reader(&self) -> io::Result<VideoReader> {
        VideoReader::new(&self.path, self.width*self.height)
    }
}

impl VideoReader {
    fn new<P: AsRef<Path>>(path: P, frame_size: usize) -> io::Result<Self> {
        let proc = cmd!(
            "ffmpeg",
            "-i", path.as_ref(),
            "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-"
        ).reader()?;

        Ok(Self {
            proc,
            frame_size,
            finished: false
        })
    }
}

impl Iterator for VideoReader {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }

        let mut buf = vec![0u8; self.frame_size];
        let ret = self.proc.read_exact(&mut buf);

        if let Err(e) = ret {
            self.finished = true;

            if e.kind() == io::ErrorKind::UnexpectedEof { None }
            else { Some(Err(e)) }
        }
        else { Some(Ok(buf)) }
    }
}
