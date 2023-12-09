use serde::{Deserialize, Serialize};

use crate::neuron::Neuron;

#[derive(Clone, Debug, Default)]
pub struct InputLayer {
	pub activations: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ActivationLayer {
	pub neurons: Vec<Neuron>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OutputLayer {
	pub neurons: Vec<Neuron>,
}

impl InputLayer {
	pub fn new(neurons_amount: usize) -> InputLayer {
		InputLayer {
			activations: Vec::with_capacity(neurons_amount),
		}
	}
}

impl ActivationLayer {
	pub fn new(neurons_amount: usize, weights_amount: usize) -> ActivationLayer {
		let mut neurons_buf = Vec::with_capacity(neurons_amount);

		for _ in 0..neurons_amount {
			neurons_buf.push(Neuron::new(weights_amount));
		}

		ActivationLayer {
			neurons: neurons_buf,
		}
	}
}

impl OutputLayer {
	pub fn new(neurons_amount: usize, weights_amount: usize) -> OutputLayer {
		let mut neurons_buf = Vec::with_capacity(neurons_amount);

		for _ in 0..neurons_amount {
			neurons_buf.push(Neuron::new(weights_amount));
		}

		OutputLayer {
			neurons: neurons_buf,
		}
	}
}
