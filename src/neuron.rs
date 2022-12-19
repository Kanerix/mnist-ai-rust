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
    pub fn new(weights_amount: usize) -> Neuron {
        let mut weights_buf = Vec::with_capacity(weights_amount);

        for _ in 0..weights_amount {
            weights_buf.push(rand::random());
        }

        Neuron {
            activation: 0.,
            weights: weights_buf,
            bias: rand::random(),
        }
    }

    pub fn from_image(img_path: &Path, size: (u32, u32)) -> Result<Neuron> {
        // Filter out the last column of the image, which is the bias
        let weights = ImageReader::open(img_path)?
            .decode()?
            .into_luma8()
            .pixels()
            .filter(|x| x.0[0] < (size.0 * size.1 - size.0) as u8)
            .map(|pixel| pixel[0] as f32 / 255.)
            .collect();

        let bias = ImageReader::open(img_path)?
            .decode()?
            .into_luma8()
            .pixels()
            .filter(|x| x.0[0] == (size.0 * size.1 - size.0) as u8)
            .map(|pixel| pixel[0] as f32 / 255.)
            .next()
            .ok_or(anyhow::anyhow!("No bias found in image"))?;

        Ok(Neuron {
            activation: 0.,
            weights,
            bias,
        })
    }

    pub fn save_as_image(&self, img_path: &Path, size: (u32, u32)) -> Result<()> {
        ImageBuffer::from_fn(size.0, size.1, |x, y| {
			println!("x: {}, y: {} y*x: {}", x, y, y*size.0 + x);
            Luma([(self.weights[(y * size.0 + x) as usize] * 255.) as u8])
        })
        .save(img_path)?;

        Ok(())
    }
}
