use log::info;
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::{
    layers::{ActivationLayer, InputLayer, OutputLayer},
    util::{
        activation_functions::{sigmoid, sigmoid_deriv},
        get_label_desc,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    pub fn train_network(&mut self) {
        let mut iterations = 0;
        let mut accuracy = 0.;

        let mut total_cost = Vec::with_capacity(60_000);
        let mut total_correct = 0;

        for (idx, (img, img_label)) in self
            .training_imgs
            .outer_iter()
            .zip(self.training_labels.outer_iter())
            .enumerate()
        {
            /* info!(
                "Training on image {} - label {}({})",
                idx,
                get_label_desc(img_label[0]),
                img_label[0]
            ); */

            let img_raw = Array1::from_shape_fn(img.len(), |x| {
                let imf_buf = img.into_shape((1, 784)).unwrap();
                imf_buf.get((0, x)).unwrap().to_owned()
            });
            self.input_layer.input_neurons = img_raw.to_vec();

            let mut cost = 0.;

            'calculate_cost: {
                let mut highest_activation = (0, 0.);

                for (idx, neuron) in self.output_layer.neurons.iter().enumerate() {
                    if neuron.activation > highest_activation.1 {
                        highest_activation = (idx, neuron.activation);
                    }

                    if idx as u8 == img_label[0] {
                        cost += f32::powf(2., neuron.activation - 1.);
                    } else {
                        cost += f32::powf(2., neuron.activation - 0.);
                    }
                }

                if highest_activation.1 == img_label[0] as f32 {
                    total_correct += 1;
                }

                break 'calculate_cost;
            }

            total_cost.push(cost);

            'back_propagate: {
                let mut output_layer_changes =
                    Array2::from_shape_vec((17, 10), vec![0.; 170]).unwrap();

                for (j, neuron) in self.output_layer.neurons.iter().enumerate() {
                    for (k, (weight, connected_neuron)) in neuron
                        .weights
                        .iter()
                        .zip(self.activation_layers.1.neurons.iter())
                        .enumerate()
                    {
                        let mut desire = 0.;

                        if j == img_label[0] as usize {
                            desire = 1.;
                        }

                        output_layer_changes[[j, k]] = connected_neuron.activation
                            * sigmoid_deriv(neuron.activation)
                            * (2. * (neuron.activation - desire));
                    }
                }

                break 'back_propagate;
            };
        }

        let total_cost_avg = total_cost.iter().sum::<f32>() / total_cost.len() as f32;
        accuracy = total_correct as f32 / 60_000.;

        info!(
            "Training complete – Cost avg: {:?} – Accuracy: {}({})",
            total_cost_avg, accuracy, total_correct
        );
    }

    /// Feed an image through the network.
    pub fn feed_forward(&self) -> (usize, f32) {
        for neuron in self.activation_layers.0.neurons.iter_mut() {
            let mut sum: f32 = 0.;
			let mut (input_layer_to_)in self.input_layer.input_neurons.iter().zip(
			)

            for (weight, activation)  {
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

		// Neuron index, activation
		let mut highest_activation: Option<(usize, f32)>;

		for (idx, neuron) in self.output_layer.neurons.iter().enumerate() {
			highest_activation = match highest_activation {
				None => Some((idx, neuron.activation)),
				Some((current_idx, current_activastion)) => {
					if neuron.activation > current_activastion {
						Some((idx, neuron.activation))
					} else {
						Some((current_idx, current_activastion))
					}
				}
			};
		}

		// Unwrap is fine, only fails if there are no neurons in the output layer.
		highest_activation.unwrap()
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
