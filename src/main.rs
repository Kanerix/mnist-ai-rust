mod layers;
mod network;
mod neuron;
mod utils;

use std::io::Write;

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
        .training_set_length(50_000)
        .test_set_length(10_000)
        .finalize();

    let training_images = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    let training_labels = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as u8);

    let test_images = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
    let test_labels = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as u8);

    let mut network = Network::new(
        training_images,
        training_labels,
        test_images,
        test_labels,
		0.1,
        (784, 16, 16, 10),
    );

	network.load_network("network_random.json");

    std::io::stdout().write(b"TRAIN/test?\n").unwrap();
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    match input.as_str().trim() {
        "train" | "" => {
            network.train_network();
        }
        "test" => {
            network.test_network();
        }
        _ => {
            panic!("Invalid input: {:?}", input.as_bytes());
        }
    }
}
