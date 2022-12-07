mod network;
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

	let training_images = Array3::from_shape_vec((60_000, 28, 28), trn_img)
		.expect("Error converting images to Array3 struct")
		.map(|x| *x as f32 / 256.0);
	let training_labels = Array2::from_shape_vec((60_000, 1), trn_lbl)
		.expect("Error converting training labels to Array2 struct")
		.map(|x| *x as f32);
	
	let _test_images = Array3::from_shape_vec((10_000, 28, 28), tst_img)
		.expect("Error converting images to Array3 struct")
		.map(|x| *x as f32 / 256.);
	let _test_labels = Array2::from_shape_vec((10_000, 1), tst_lbl)
		.expect("Error converting testing labels to Array2 struct")
		.map(|x| *x as f32);

	let mut _network = Network::new(training_images, training_labels);
}