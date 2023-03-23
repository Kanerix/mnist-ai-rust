use serde::{Serialize, Deserialize};

use crate::neuron::Neuron;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputLayer {
    pub input_neurons: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationLayer {
    pub neurons: Vec<Neuron>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputLayer {
    pub neurons: Vec<Neuron>,
}

impl InputLayer {
    pub fn new(neurons_len: usize) -> InputLayer {
        InputLayer {
            input_neurons: Vec::with_capacity(neurons_len),
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
