use log::info;
use ndarray::{Array3, Array2, Array1};

use crate::{layers::{InputLayer, ActivationLayer, OutputLayer}, util::{get_label_desc, save_image}};

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
		shape: (usize, usize, usize, usize)
	) -> Network {
		Network {
			trn_imgs,
			trn_lbls,
			test_imgs,
			test_lbls,
			input_layer: InputLayer::new(shape.0),
			activation_layers: ( 
				ActivationLayer::new(shape.1),
				ActivationLayer::new(shape.2)
			),
			output_layer: OutputLayer::new(shape.3),
		}
	}

	pub fn train_network(&mut self) {
		for ((idx, img), lbl) in self.trn_imgs
			.outer_iter()
			.enumerate()
			.zip(self.trn_lbls.outer_iter())
		{
			info!("Training on image {} which is: {}", idx, get_label_desc(lbl[0]));

			let img_raw = Array1::from_shape_fn(img.len(), |x| {
				let imf_buf = img.into_shape((1, 784)).unwrap();
				imf_buf.get((0, x)).unwrap().to_owned()
			});
			self.input_layer.neurons = img_raw.to_vec();

			'feed_forward: {
				break 'feed_forward;
			};

			'back_propagate: {
				break 'back_propagate;
			};
		}

		info!("Training complete");
	}

	fn save_network_state() {
		todo!();
	}
}
