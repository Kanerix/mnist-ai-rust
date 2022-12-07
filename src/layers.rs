use crate::neuron::Neuron;


pub trait Layer {
	fn calculate_cost(&self, expected: &Vec<f32>) -> f32;
}


pub struct InputLayer {
	pub neurons: Vec<f32>,
}

impl InputLayer {
	pub fn new(neurons_len: usize) -> InputLayer {
		InputLayer {
			neurons: vec![0.0; neurons_len],
		}
	}
}


pub struct ActivationLayer {
	pub neurons: Vec<Neuron>,
}

impl ActivationLayer {
	pub fn new(neurons_len: usize) -> ActivationLayer {
		ActivationLayer {
			neurons: vec![Neuron::new(neurons_len); 16],
		}
	}
}


pub struct OutputLayer {
	pub neurons: Vec<Neuron>,
}

impl OutputLayer {
	pub fn new(neurons_len: usize) -> OutputLayer {
		OutputLayer {
			neurons: vec![Neuron::new(neurons_len); 16],
		}
	}
}