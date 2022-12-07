use ndarray::{Array3, Array2};

pub struct Network {
	pub training_images: Array3<f32>,
	pub training_labels: Array2<f32>,
	pub input_layer: Vec<f32>,
	pub activation_layers: Vec<ActivationLayer>,
	pub output_layer: Vec<u8>,
}

impl Network {
	pub fn new(training_images: Array3<f32>, training_labels: Array2<f32>) -> Network {
		Network {
			training_images,
			training_labels,
			input_layer: Vec::with_capacity(784),
			activation_layers: Vec::with_capacity(2),
			output_layer: Vec::with_capacity(10),
		}
	}
}

pub struct ActivationLayer {
	pub neurons: Vec<Neuron>,


}

pub struct Neuron {
	pub weights: Vec<f64>,
	pub bias: f64,
}