mod network;
mod layers;
mod neuron;
mod util;

use network::Network;
use log4rs;
use mnist::*;
use ndarray::prelude::*;

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

	let trn_imgs = Array3::from_shape_vec((60_000, 28, 28), trn_img)
		.expect("Error converting images to Array3 struct")
		.map(|x| *x as f32 / 256.0);
	let trn_lbls = Array2::from_shape_vec((60_000, 1), trn_lbl)
		.expect("Error converting training labels to Array2 struct")
		.map(|x| *x as u8);
	
	let test_imgs = Array3::from_shape_vec((10_000, 28, 28), tst_img)
		.expect("Error converting images to Array3 struct")
		.map(|x| *x as f32 / 256.);
	let test_lbls = Array2::from_shape_vec((10_000, 1), tst_lbl)
		.expect("Error converting testing labels to Array2 struct")
		.map(|x| *x as u8);

	let mut network = Network::new(
		trn_imgs,
		trn_lbls,
		test_imgs,
		test_lbls,
		(784, 16, 16, 10)
	);

	network.train_network();
}