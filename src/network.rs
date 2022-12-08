use std::path::Path;

use anyhow::Result;
use log::info;
use ndarray::{Array1, Array2, Array3};

use crate::{
    layers::{ActivationLayer, InputLayer, OutputLayer},
    neuron::Neuron,
    util::{activation_functions::sigmoid, get_label_desc},
};

pub struct Network {
    pub trn_imgs: Array3<f32>,
    pub trn_lbls: Array2<u8>,
    pub test_imgs: Array3<f32>,
    pub test_lbls: Array2<u8>,
    pub input_layer: InputLayer,
    pub activation_layers: (ActivationLayer, ActivationLayer),
    pub output_layer: OutputLayer,
}

impl Network {
    pub fn new(
        trn_imgs: Array3<f32>,
        trn_lbls: Array2<u8>,
        test_imgs: Array3<f32>,
        test_lbls: Array2<u8>,
        shape: (usize, usize, usize, usize),
    ) -> Network {
        Network {
            trn_imgs,
            trn_lbls,
            test_imgs,
            test_lbls,
            input_layer: InputLayer::new(shape.0),
            activation_layers: (ActivationLayer::new(shape.1), ActivationLayer::new(shape.2)),
            output_layer: OutputLayer::new(shape.3),
        }
    }

    pub fn train_network(mut self) {
        for ((idx, img), lbl) in self
            .trn_imgs
            .outer_iter()
            .enumerate()
            .zip(self.trn_lbls.outer_iter())
        {
            info!(
                "Training on image {} which is: {}",
                idx,
                get_label_desc(lbl[0])
            );

            let img_raw = Array1::from_shape_fn(img.len(), |x| {
                let imf_buf = img.into_shape((1, 784)).unwrap();
                imf_buf.get((0, x)).unwrap().to_owned()
            });
            self.input_layer.input_neurons = img_raw.to_vec();

            'feed_forward: {
                // From input to first hidden layer
                for neuron in self.activation_layers.0.neurons.iter_mut() {
                    let mut sum: f32 = 0.;

                    for (weight, activation) in neuron
                        .weights
                        .iter()
                        .zip(self.input_layer.input_neurons.iter())
                    {
                        sum += weight * activation;
                    }

                    info!("sum: {}", sum);

                    neuron.activation = sigmoid(sum) - neuron.bias;
                }

                // From first hidden layer to second hidden layer
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

                    neuron.activation = sigmoid(sum) - neuron.bias;
                }

                // From second hidden layer to output layer
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

                    neuron.activation = sigmoid(sum - neuron.bias);
                }

                let mut guess: (u8, f32) = (0, 0.);

                for (idx, neuron) in self.output_layer.neurons.iter().enumerate() {
                    if neuron.activation > guess.1 {
                        guess = (idx as u8, neuron.activation);
                    }
                }

                info!(
                    "Feed forward complete. Guess is {} – Activation: {} – Correct: {}",
                    get_label_desc(guess.0 as u8),
                    guess.1,
                    get_label_desc(lbl[0])
                );

                break 'feed_forward;
            };

            if idx == 1 {
                break;
            }

            'back_propagate: {
                break 'back_propagate;
            };

            self.save_network_as_images().unwrap();
        }

        info!("Training complete");
    }

    pub fn test_network(&self) {
        todo!();
    }

    pub fn save_network_as_images(&self) -> Result<()> {
        for (idx, neuron) in self.activation_layers.0.neurons.iter().enumerate() {
            neuron.save_as_image(
                Path::new(&format!("activation_layer_1_neuron_{idx}.png")),
                (28, 28),
            )?;
        }

        for (idx, neuron) in self.activation_layers.1.neurons.iter().enumerate() {
            neuron.save_as_image(
                Path::new(&format!("activation_layer_2_neuron_{idx}.png")),
                (4, 4),
            )?;
        }

        for (idx, neuron) in self.output_layer.neurons.iter().enumerate() {
            neuron.save_as_image(Path::new(&format!("output_layer_neuron_{idx}.png")), (4, 4))?;
        }

        Ok(())
    }

    pub fn load_network_from_images(&mut self) -> Result<()> {
        for (idx, neuron) in self.activation_layers.0.neurons.iter_mut().enumerate() {
            *neuron =
                Neuron::from_image(Path::new(&format!("activation_layer_1_neuron_{idx}.png")))?;
        }

        for (idx, neuron) in self.activation_layers.1.neurons.iter_mut().enumerate() {
            *neuron =
                Neuron::from_image(Path::new(&format!("activation_layer_2_neuron_{idx}.png")))?;
        }

        for (idx, neuron) in self.output_layer.neurons.iter_mut().enumerate() {
            *neuron = Neuron::from_image(Path::new(&format!("output_layer_neuron_{idx}.png")))?;
        }

        Ok(())
    }
}
