use log::info;
use ndarray::{Array1, Array2, Array3};

use crate::{
    layers::{ActivationLayer, InputLayer, OutputLayer},
    util::{activation_functions::sigmoid, get_label_desc},
};

pub struct Network {
    pub training_imgs: Array3<f32>,
    pub training_labels: Array2<u8>,
    pub test_imgs: Array3<f32>,
    pub test_labels: Array2<u8>,
    pub input_layer: InputLayer,
    pub activation_layers: (ActivationLayer, ActivationLayer),
    pub output_layer: OutputLayer,
}

impl Network {
    pub fn new(
        training_imgs: Array3<f32>,
        training_labels: Array2<u8>,
        test_imgs: Array3<f32>,
        test_labels: Array2<u8>,
        shape: (usize, usize, usize, usize),
    ) -> Network {
        Network {
            training_imgs,
            training_labels,
            test_imgs,
            test_labels,
            input_layer: InputLayer::new(shape.0),
            activation_layers: (
                ActivationLayer::new(shape.1, shape.0),
                ActivationLayer::new(shape.2, shape.1),
            ),
            output_layer: OutputLayer::new(shape.3, shape.2),
        }
    }

    pub fn train_network(mut self) {
		let mut total_cost = Vec::with_capacity(60_000);

        for (idx, (img, img_label)) in self
            .training_imgs
            .outer_iter()
            .zip(self.training_labels.outer_iter())
            .enumerate()
        {
            info!(
                "Training on image {} - label {}({})",
                idx,
                get_label_desc(img_label[0]),
                img_label[0]
            );

            let img_raw = Array1::from_shape_fn(img.len(), |x| {
                let imf_buf = img.into_shape((1, 784)).unwrap();
                imf_buf.get((0, x)).unwrap().to_owned()
            });
            self.input_layer.input_neurons = img_raw.to_vec();

            'feed_forward: {
                for neuron in self.activation_layers.0.neurons.iter_mut() {
                    let mut sum: f32 = 0.;

                    for (weight, activation) in neuron
                        .weights
                        .iter()
                        .zip(self.input_layer.input_neurons.iter())
                    {
                        sum += weight * activation;
                    }

                    neuron.activation = sigmoid(sum + (neuron.bias));
                }

                for neuron in self.activation_layers.1.neurons.iter_mut() {
                    let mut sum: f32 = 0.;

                    for (weight, activation) in neuron.weights.iter().zip(
                        self.activation_layers
                            .0
                            .neurons
                            .iter()
                            .map(|x| x.activation),
                    ) {
                        sum += weight * activation;
                    }

                    neuron.activation = sigmoid(sum + (neuron.bias));
                }

                for neuron in self.output_layer.neurons.iter_mut() {
                    let mut sum: f32 = 0.;

                    for (weight, activation) in neuron.weights.iter().zip(
                        self.activation_layers
                            .1
                            .neurons
                            .iter()
                            .map(|x| x.activation),
                    ) {
                        sum += weight * activation;
                    }

                    neuron.activation = sigmoid(sum + (neuron.bias));
                }

                info!(
                    "Output activations: {:?}",
                    self.output_layer
                        .neurons
                        .iter()
                        .map(|x| x.activation)
                        .collect::<Vec<f32>>()
                );

                break 'feed_forward;
            };

            let mut cost = 0.;

            'calculate_cost: {
                for (idx, neuron) in self.output_layer.neurons.iter().enumerate() {
                    if idx as u8 == img_label[0] {
                        cost += f32::powf(2., neuron.activation - 1.);
                    } else {
                        cost += f32::powf(2., neuron.activation - 0.);
                    }
                }

				info!("Cost: {:?}", cost);

                break 'calculate_cost;
            }

			total_cost.push(cost);

            'back_propagate: {
				// Calculate output layer deltas
				
				break 'back_propagate;
            };
        }

		let total_cost_avg = total_cost.iter().sum::<f32>() / total_cost.len() as f32;

        info!("Training complete: Cost avg {:?}", total_cost_avg);
    }

    pub fn test_network(&self) {
        for (idx, (_img, img_label)) in self
            .test_imgs
            .outer_iter()
            .zip(self.test_labels.outer_iter())
            .enumerate()
        {
            info!(
                "Testing on image {} - label {}({})",
                idx,
                get_label_desc(img_label[0]),
                img_label[0]
            );
        }
    }
}
