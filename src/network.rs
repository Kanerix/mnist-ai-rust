use std::{fs::File, io::{Write, Read}};

use log::info;
use mnist::{Mnist, MnistBuilder};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::{
    layers::{ActivationLayer, InputLayer, OutputLayer},
    utils::{activation_functions::sigmoid, save_neuron_as_image},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Network {
    // Skip serializing fields that are not needed for the network to work.
    #[serde(skip_serializing, skip_deserializing)]
    pub training_images: Box<Array3<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub training_labels: Box<Array2<u8>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub test_images: Box<Array3<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub test_labels: Box<Array2<u8>>,

	pub learning_rate: f32,

    // All the layers of the network.
    pub input_layer: InputLayer,
    pub activation_layers: (ActivationLayer, ActivationLayer),
    pub output_layer: OutputLayer,
}

impl Network {
    pub fn new(
		learning_rate: f32,
        shape: (usize, usize, usize, usize),
    ) -> Network {
        // Setup the dataset and allocate it on the heap.
		let Mnist {
			trn_img,
			trn_lbl,
			tst_img,
			tst_lbl,
			..
		} = MnistBuilder::new()
			.label_format_digit()
			.training_set_length(50_000)
			.test_set_length(10_000)
			.finalize();

		let training_images = Box::new(Array3::from_shape_vec((50_000, 28, 28), trn_img)
			.expect("Error converting images to Array3 struct")
			.map(|x| *x as f32 / 256.0));
		let training_labels = Box::new(Array2::from_shape_vec((50_000, 1), trn_lbl)
			.expect("Error converting training labels to Array2 struct")
			.map(|x| *x as u8));

		let test_images = Box::new(Array3::from_shape_vec((10_000, 28, 28), tst_img)
			.expect("Error converting images to Array3 struct")
			.map(|x| *x as f32 / 256.));
		let test_labels = Box::new(Array2::from_shape_vec((10_000, 1), tst_lbl)
			.expect("Error converting testing labels to Array2 struct")
			.map(|x| *x as u8));

        // Return a new network.
        Network {
            training_images,
            training_labels,
            test_images,
            test_labels,
			learning_rate,
            input_layer: InputLayer::new(shape.0),
            activation_layers: (
                ActivationLayer::new(shape.1, shape.0),
                ActivationLayer::new(shape.2, shape.1),
            ),
            output_layer: OutputLayer::new(shape.3, shape.2),
        }
    }

    pub fn train(&mut self) {
        // How many times have we gone through the entire dataset.
        let mut dataset_iteration = 0;

        while dataset_iteration < 100 {
            // An array to store the cost of each iteration.
            let mut cost_array = Vec::with_capacity(self.training_images.len());
            // Stores how many images the network has gotten correct.
            let mut correct_images = 0;

            let images = self.training_images.clone();
            let labels = self.training_labels.clone();
            let outer_images = images.outer_iter();
            let outer_labels = labels.outer_iter();

            // Loop that runs over all training images.
            for (_idx, (image, image_label)) in outer_images.zip(outer_labels).enumerate() {
                // Make the image pixels into a 1D array for the input layer.
                let image_buffer = image.into_shape((1, 784)).unwrap();
                let raw_image = image_buffer.iter().map(|x| *x).collect();

                // Feed the current image forward through the network.
                self.feed_forward(raw_image);

                // Get the neuron with the highest activation.
                let neuron = self.get_most_active_neuron().unwrap();
                if neuron.0 == image_label[0] as usize {
                    correct_images += 1;
                }

                // Get the cost after an images has been fed forward.
                cost_array.push(self.calculate_iteration_cost(image_label[0]));

                // Back propagate the error.
                self.back_propagate(image_label[0]);
            }

            dataset_iteration += 1;
			// Calculate the accuracy of the network.
            let accuracy = ((correct_images as f32) / 50_000.) * 100.;
			// Calculate the average cost of the entire dataset.
			let avg_cost = cost_array.iter().sum::<f32>() / cost_array.len() as f32;
            info!(
                "Dataset iteration {} complete – Accuracy: {}% ({}), avg. cost: {}",
                dataset_iteration, accuracy, correct_images, avg_cost
            );
        }

		self.save_layers("network.json").unwrap();
    }

    pub fn test(&mut self) {
		let images = self.test_images.clone();
		let labels = self.test_labels.clone();
        let outer_images = images.outer_iter();
        let outer_labels = labels.outer_iter();

        // An array to store the cost of each iteration.
		let mut cost_array = Vec::with_capacity(self.test_images.len());
		// Stores how many images the network has gotten correct.
		let mut correct_images = 0;

        for (image, image_label) in outer_images.zip(outer_labels) {
			// Make the image pixels into a 1D array for the input layer.
			let image_buffer = image.into_shape((1, 784)).unwrap();
			let raw_image = image_buffer.iter().map(|x| *x).collect();

			// Feed the current image forward through the network.
			self.feed_forward(raw_image);

			// Get the neuron with the highest activation.
			let neuron = self.get_most_active_neuron().unwrap();
			if neuron.0 == image_label[0] as usize {
                correct_images += 1;
			}

			// Get the cost after an images has been fed forward.
			cost_array.push(self.calculate_iteration_cost(image_label[0]));
		}

		// Calculate the accuracy of the network.
		let accuracy = ((correct_images as f32) / 50_000.) * 100.;
		// Calculate the average cost of the entire dataset.
		let avg_cost = cost_array.iter().sum::<f32>() / cost_array.len() as f32;
		info!(
			"Test complete – Accuracy: {}% ({}), avg. cost: {}",
			accuracy, correct_images, avg_cost
		);

		for (i, neuron) in self.activation_layers.0.neurons.iter().enumerate() {
			save_neuron_as_image(neuron, &format!("neuron_{}.png", i));
		}
	}

    /// Feed an raw image through the network, and update all the neuron activations.
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

	/// Back propagate the error through the network. 
	/// Label is the correct label for the current image.
    pub fn back_propagate(&mut self, label: u8) {
		// Calulate the desired output for the network.
        let mut desire = Vec::with_capacity(self.output_layer.neurons.len());
		for i in 0..self.output_layer.neurons.len() {
			if i == label as usize {
				desire.push(1.);
			} else {
				desire.push(0.);
			}
		}

		// Calculate the error signal for the output layer.
        let mut output_error_signal = Vec::with_capacity(self.output_layer.neurons.len());
        for (i, neuron) in self.output_layer.neurons.iter().enumerate() {
			// The label is the desired output for the network.
            output_error_signal.push(desire[i] - neuron.activation); 
        }

        // Update the weights and bias for the output layer.
        for (i, output_neuron) in self.output_layer.neurons.iter_mut().enumerate() {
            for j in 0..self.activation_layers.1.neurons.len() {
				let previous_activation = output_neuron.activation;
                let delta_weight = self.learning_rate * output_error_signal[i] * previous_activation;

                output_neuron.weights[j] += delta_weight;
            }

            let delta_bias = self.learning_rate * output_error_signal[i];
            output_neuron.bias += delta_bias;
        }

        // Calculate the error signal for the second activation layer
		let mut second_activation_error_signal = Vec::with_capacity(self.activation_layers.1.neurons.len());
        for (i, second_activation_neuron) in self.activation_layers.1.neurons.iter().enumerate() {
            let mut error_sum = 0.0;
            for j in 0..self.output_layer.neurons.len() {
                error_sum += output_error_signal[j] * self.output_layer.neurons[j].weights[i];
            }

            second_activation_error_signal.push(
				error_sum * second_activation_neuron.activation *
				(1.0 - second_activation_neuron.activation)
			);
        }

        // Update the weights and bias for the second activation layer
        for (i, activation_neuron) in self.activation_layers.1.neurons.iter_mut().enumerate() {
            for j in 0..self.activation_layers.0.neurons.len() {
				let previoues_activation = self.activation_layers.0.neurons[j].activation;
                let delta_weight = self.learning_rate * second_activation_error_signal[i] * previoues_activation;

                activation_neuron.weights[j] += delta_weight;
            }

            let delta_bias = self.learning_rate * second_activation_error_signal[i];
            activation_neuron.bias += delta_bias;
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

	/// Save the network layers, needed to restore the networks state, to a json file.
	pub fn save_layers(&self, name: impl Into<String>) -> anyhow::Result<()> {
		let name_into = name.into();
		// Save the network to the "networks" folder.
		let mut file = File::create(&format!("networks/{}.json", name_into))?;
		let json = serde_json::to_string(&self)?;
		file.write_all(json.as_bytes())?;	
		Ok(())
	}

	/// Load the network layers, needed to restore the networks state, from a json file.
	pub fn load_layers(&mut self, name: impl Into<String>) -> anyhow::Result<()> {
		let name_into: String = name.into();
		// Load the file from the "networks" folder.
		let mut file = File::open(&format!("networks/{}.json", name_into))?;
		let mut contents = String::new();
		file.read_to_string(&mut contents)?;
		let loaded: Network = serde_json::from_str(&contents)?;

		// Set the networks layers to the loaded layers.
		self.learning_rate = loaded.learning_rate;
		self.activation_layers = loaded.activation_layers;
		self.output_layer = loaded.output_layer;
		Ok(())
	}
}
