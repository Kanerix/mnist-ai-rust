use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub activation: f32,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    pub fn new(weights_amount: usize) -> Neuron {
        let mut weights_buf = Vec::with_capacity(weights_amount);

        for _ in 0..weights_amount {
            weights_buf.push(rand::random::<f32>() * 2. - 1.);
        }

        Neuron {
            activation: 0.,
            weights: weights_buf,
            bias: rand::random(),
        }
    }
}
