use log::info;
use ndarray::{Array1, Array2, Array3};

use crate::{
    layers::{ActivationLayer, InputLayer, OutputLayer},
    util::activation_functions::{sigmoid, sigmoid_deriv},
};

#[derive(Clone, Debug)]
pub struct Network {
    pub training_images: Array3<f32>,
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
            training_images: training_imgs,
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
        // How many times have we gone through the entire dataset.
        let mut dataset_iteration = 0;

        while dataset_iteration < 10 {
            let outer_images = self.training_images.outer_iter();
            let outer_labels = self.training_labels.outer_iter();

            // Loop that runs over all training images.
            for (idx, (image, image_label)) in outer_images.zip(outer_labels).enumerate() {
                // Make the image into a 1D array for the input layer.
                let raw_image = Array1::from_shape_fn(image.len(), |x| {
                    let imf_buf = image.into_shape((1, 784)).unwrap();
                    imf_buf.get((0, x)).unwrap().to_owned()
                }).to_vec();

				// Feed the current image forward through the network.
                self.feed_forward(raw_image); // Box pointer yes
                let cost = self.calculate_iteration_cost(image_label[0]);

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

                            if j == image_label[0] as usize {
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

            dataset_iteration += 1;

            info!(
                "Dataset iteration {} complete – Accuracy: {}({})",
                dataset_iteration, 0., 0.
            );
        }
    }

    /// Feed an image through the network, and update all the neuron activations.
    pub fn feed_forward(&mut self, raw_image: Vec<f32>) {
        self.input_layer.activations = raw_image;

        // From input layer to the first activation layer.
        for neuron in self.activation_layers.0.neurons.iter_mut() {
            let mut sum: f32 = 0.;
            let weights = neuron.weights.iter();
            let activations = self.input_layer.activations.iter();

            for (weight, activation) in weights.zip(activations) {
                sum += weight * activation;
            }

            neuron.activation = sigmoid(sum + (neuron.bias));
        }

        // From activation layer 1 to activation layer 2.
        for current_neuron in self.activation_layers.1.neurons.iter_mut() {
            let mut sum: f32 = 0.;
            let wieghts = current_neuron.weights.iter();
            let previous_neurons = self.activation_layers.0.neurons.iter();

            for (weight, neuron) in wieghts.zip(previous_neurons) {
                sum += weight * neuron.activation;
            }

            current_neuron.activation = sigmoid(sum + (current_neuron.bias));
        }

        // From activation layer 2 to output layer.
        for output_neuron in self.output_layer.neurons.iter_mut() {
            let mut sum: f32 = 0.;
            let weights = output_neuron.weights.iter();
            let previous_neurons = self.activation_layers.1.neurons.iter();

            for (weight, neuron) in weights.zip(previous_neurons) {
                sum += weight * neuron.activation;
            }

            output_neuron.activation = sigmoid(sum + (output_neuron.bias));
        }
    }

    /// Get the most active neuron in the output layer for the current dataset iteration.
    /// Returns "None" if there is no neurons in the output layer.
    pub fn get_most_active_neuron(&self) -> Option<(usize, f32)> {
		// Stores the most active neuron in an option tuple.
        let mut most_active_output_neuron: Option<(usize, f32)> = None;

        // Loop through all output neurons.
        for (index, neuron) in self.output_layer.neurons.iter().enumerate() {
            most_active_output_neuron = match most_active_output_neuron {
                // If there is no most active neuron, set it to the current neuron.
                None => Some((index, neuron.activation)),
                // If there is a most active neuron, check if the current neuron is more active.
                Some((current_idx, current_activastion)) => {
                    if neuron.activation > current_activastion {
                        Some((index, neuron.activation))
                    } else {
                        Some((current_idx, current_activastion))
                    }
                }
            };
        }

        most_active_output_neuron
    }

	/// Calculate the cost of the current image iteration.
    pub fn calculate_iteration_cost(&self, label: u8) -> f32 {
		// Stores the cost of the current iteration in a float.
        let mut cost: f32 = 0.;

        for (index, neuron) in self.output_layer.neurons.iter().enumerate() {
			if index as u8 == label {
				cost += f32::powf(2., neuron.activation - 1.);
			} else {
				cost += f32::powf(2., neuron.activation - 0.);
			}
        }

		cost 
    }

    pub fn test_network(&self) {
        let outer_images = self.training_images.outer_iter();
        let outer_labels = self.training_labels.outer_iter();

        for (_idx, (_image, _image_label)) in outer_images.zip(outer_labels).enumerate() {
            todo!("Test network with the test dataset.")
        }
    }
}
