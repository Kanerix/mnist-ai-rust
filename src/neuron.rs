use image::{self, ImageBuffer, Luma};

#[derive(Clone)]
pub struct Neuron {
	pub weights: Vec<f64>,
	pub bias: f64,
}

impl Neuron {
	pub fn new(wghts_len: usize) -> Neuron {
		Neuron {
			weights: vec![rand::random(); wghts_len],
			bias: 0.0,
		}
	}

	pub fn save_state_as_image(&self, name: &str) {
		ImageBuffer::from_fn(28, 28, |x, y| {
			Luma([
				(self.weights[y as usize * 28 + x as usize] * 256.0) as u8
			])
		}).save(format!("neurons/{name}.png")).unwrap();
	}
}