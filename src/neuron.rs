use std::path::Path;

use anyhow::Result;
use image::{self, io::Reader as ImageReader, ImageBuffer, Luma};

#[derive(Clone, Debug)]
pub struct Neuron {
    pub activation: f32,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(wghts_len: usize) -> Neuron {
        let mut wghts_buf = Vec::with_capacity(wghts_len);

        for i in 0..wghts_len {
            wghts_buf[i] = rand::random::<f32>() * 2.0 - 1.0;
        }

        Neuron {
            activation: 0.,
            weights: wghts_buf,
            bias: 0.,
        }
    }

    pub fn from_image(img_path: &Path) -> Result<Neuron> {
        let weights = ImageReader::open(img_path)?
            .decode()?
            .to_luma8()
            .pixels()
            .map(|p| p[0] as f32 / 256.0)
            .collect::<Vec<f32>>();

        Ok(Neuron {
            activation: 0.0,
            weights,
            bias: 0.2,
        })
    }

    pub fn save_as_image(&self, img_path: &Path, size: (u32, u32)) -> Result<()> {
        ImageBuffer::from_fn(size.0, size.1, |x, y| {
			Luma([self.weights[(y * size.0 + x) as usize] as u8])
        })
        .save(img_path)?;

        Ok(())
    }
}
