use image::{self, ImageBuffer, Luma};
use ndarray::Array3;

#[allow(dead_code)]
pub fn get_label_desc(label: impl Into<u8>) -> String {
    let desc = match label.into() {
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
    };

    desc.to_string()
}

#[allow(dead_code)]
pub fn save_image(image_num: usize, image_data: &Array3<f32>) {
    ImageBuffer::from_fn(28, 28, |x, y| {
        Luma([(image_data[[image_num, y as usize, x as usize]] * 256.0) as u8])
    })
    .save(format!("images/{image_num}.png"))
    .unwrap();
}

#[allow(dead_code)]
pub mod activation_functions {
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}
