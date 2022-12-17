mod layers;
mod network;
mod neuron;
mod util;

use std::io::Write;

use log::{info, warn};
use log4rs;
use mnist::*;
use ndarray::prelude::*;
use network::Network;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let training_imgs = Array3::from_shape_vec((60_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    let training_labels = Array2::from_shape_vec((60_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as u8);

    let test_imgs = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
    let test_labels = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as u8);

    let mut network = Network::new(
        training_imgs,
        training_labels,
        test_imgs,
        test_labels,
        (784, 16, 16, 10),
    );

    match network.load_network_from_images() {
        Ok(_) => info!("Loaded network from images"),
        Err(_) => warn!("Did not find a local network"),
    };

	std::io::stdout().write(b"train/test?\n").unwrap();
	std::io::stdout().flush().unwrap();

	let mut input = String::new();
	std::io::stdin().read_line(&mut input).unwrap();

	match input.as_str() {
		"train\n" => {
			network.train_network();
		},
		"test\n" => {
			network.test_network();
		},
		_ => {
			panic!("Invalid input")
		}
	}
}
