use serde::{Deserialize, Serialize};

use crate::neuron::Neuron;

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
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
            neurons: neurons_buf,
        }
    }
}
