use nannou::prelude::*;
use ndarray::Array2;

use crate::network::Network;

struct Model {
	network: Network	
}

pub fn run() {
    nannou::app(model)
		.size(28*32, 28*32)
		.event(event)
		.simple_window(view)
		.run();
}

fn model(_app: &App) -> Model {
	let network = Network::new(0.1, (784,16, 16, 10));

    Model {
		network
	}
}

fn event(_app: &App, _model: &mut Model, _event: Event) {
}

fn view(app: &App, model: &Model, frame: Frame) {
	let draw = app.draw();
	draw.background().color(BLACK);

	let pixels: Array2<u8> = Array2::zeros((28, 28)); 

	for x in -13..15 {
		for y in -13..15 {
			draw.rect()
				.x_y(x as f32 * 28.0, y as f32 * 28.0)
				.w_h(32., 32.)
				.color(BLACK)
				.stroke_weight(1.)
				.stroke(WHITE);
		}
	}

	draw.to_frame(app, &frame).unwrap();
}
