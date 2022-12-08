use std::path::Path;

use anyhow::Result;
use image::{self, io::Reader as ImageReader, ImageBuffer, Luma};

const DEFAULT_BIAS: f32 = 1.;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub activation: f32,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weights_amount: usize) -> Neuron {
        let mut weights_buf = Vec::with_capacity(weights_amount);

        for _ in 0..weights_amount {
            weights_buf.push(rand::random::<f32>());
        }

        Neuron {
            activation: 0.,
            weights: weights_buf,
            bias: DEFAULT_BIAS,
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
            bias: DEFAULT_BIAS,
        })
    }

    pub fn save_as_image(&self, img_path: &Path, size: (u32, u32)) -> Result<()> {
        ImageBuffer::from_fn(size.0, size.1, |x, y| {
			Luma([(self.weights[(y * size.0 + x) as usize] * 255.) as u8])
        })
        .save(img_path)?;

        Ok(())
    }
}
