use crate::neuron::Neuron;

pub trait Layer {
    fn calculate_cost(&self, expected: &Vec<f32>) -> f32;
}

pub struct InputLayer {
    pub input_neurons: Vec<f32>,
}

impl InputLayer {
    pub fn new(neurons_len: usize) -> InputLayer {
        InputLayer {
            input_neurons: Vec::with_capacity(neurons_len),
        }
    }
}

pub struct ActivationLayer {
    pub neurons: Vec<Neuron>,
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

pub struct OutputLayer {
    pub neurons: Vec<Neuron>,
}

impl OutputLayer {
    pub fn new(neurons_amount: usize, weights_amount: usize) -> OutputLayer {
		let mut neurons_buf = Vec::with_capacity(neurons_amount);

		for _ in 0..neurons_amount {
			neurons_buf.push(Neuron::new(weights_amount));
		}

        OutputLayer {
            neurons: neurons_buf
        }
    }
}
