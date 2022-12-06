use image::{self, ImageBuffer, Luma};
use ndarray::Array3;

pub fn get_label_description(label: u8) -> impl Into<String> {
	match label {
		0 => "T-shirt/top",
		1 => "Trouser",
		2 => "Pullover",
		3 => "Dress",
		4 => "Coat",
		5 => "Sandal",
		6 => "Shirt",
		7 => "Sneaker",
		8 => "Bag",
		9 => "Ankle boot",
		_ => "Unknown",
	}
}

pub fn save_image(image_num: usize, image_data: &Array3<f32>) {
	ImageBuffer::from_fn(28, 28, |x, y| {
		Luma([
			(image_data[[image_num, y as usize, x as usize]] * 256.0) as u8
		])
	}).save(format!("images/{image_num}.png")).unwrap();
}