use clap::{command, Parser};
use mnist_ai_rust::network::Network;
use nannou::prelude::*;
use nannou_egui::{egui, Egui};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	#[arg(short, long, default_value = None)]
	input: String,
}

fn main() {
	nannou::app(model).update(update).run();
}

struct Model {
	egui: Egui,
	network: Network,
	grid: Grid,
	loaded_image: u32,
	loaded_label: u8,
}

const WIDTH: u32 = 1400;
const HEIGHT: u32 = 1000;

const ROWS: f32 = 28.;
const COLS: f32 = 28.;
const CELL_SIZE: f32 = 28.;

struct Grid {
	cells: Vec<Vec<Cell>>,
}

#[derive(Clone)]
struct Cell {
	position: Point2,
	activation: f32,
}

impl Grid {
	fn on_click(&mut self, mouse_pos: Point2) {
		let cells = self.cells.iter_mut().flatten();

		for cell in cells {
			let distance = cell.position.distance(mouse_pos);

			if distance < CELL_SIZE {
				cell.activation += distance * 0.01;
			}
		}
	}
}

impl Cell {
	fn draw(&self, draw: &Draw) {
		draw.rect()
			.x_y(self.position.x, self.position.y - CELL_SIZE)
			.w_h(CELL_SIZE, CELL_SIZE)
			.color(rgb(self.activation, self.activation, self.activation))
			.stroke(WHITE)
			.stroke_weight(1.);
	}
}

fn model(app: &App) -> Model {
	let window_id = app
		.new_window()
		.size(WIDTH, HEIGHT)
		.raw_event(raw_window_event)
		.view(view)
		.build()
		.unwrap();
	let window = app.window(window_id).unwrap();

	let args = Args::parse();

	let mut network = Network::new(0.1, (784, 16, 16, 10));
	network.load_layers(args.input).unwrap();

	let egui = Egui::from_window(&window);

	let mut grid = Grid {
		cells: Vec::with_capacity(COLS as usize),
	};

	for col in 0..COLS as usize {
		grid.cells.push(Vec::with_capacity(ROWS as usize));
		let col_row = grid.cells.get_mut(col).unwrap();

		for row in 0..ROWS as usize {
			let grid_w = COLS * CELL_SIZE;
			let grid_h = ROWS * CELL_SIZE;

			let x = (row as f32 * CELL_SIZE) - (grid_w / 2.) + (CELL_SIZE / 2.);
			let y = ((COLS - col as f32) as f32 * CELL_SIZE) - (grid_h / 2.) + (CELL_SIZE / 2.);

			col_row.push(Cell {
				position: Point2::new(x, y),
				activation: 0.,
			});
		}
	}

	Model {
		egui,
		network,
		grid,
		loaded_image: 0,
		loaded_label: 0,
	}
}

fn update(app: &App, model: &mut Model, update: Update) {
	let Model {
		ref mut egui,
		ref mut network,
		ref mut grid,
		ref mut loaded_image,
		ref mut loaded_label,
	} = *model;

	if app.mouse.buttons.left().is_down() {
		let mouse = app.mouse.position();
		grid.on_click(mouse);
	}

	network.feed_forward(
		grid.cells
			.iter()
			.flatten()
			.map(|cell| cell.activation)
			.collect(),
	);

	egui.set_elapsed_time(update.since_start);
	let ctx = egui.begin_frame();

	egui::Window::new("Output Layer")
		.default_size(egui::vec2(150., 0.))
		.show(&ctx, |ui| {
			for (index, neuron) in network.output_layer.neurons.iter().enumerate() {
				ui.label(format!("Neuron {}: {:.4}", index, neuron.activation));
			}
		});

	egui::Window::new("Output")
		.default_size(egui::vec2(150., 0.))
		.show(&ctx, |ui| {
			let most_active_neuron = network.get_most_active_neuron().unwrap();
			ui.label(format!("Guess: {}", most_active_neuron.0));
			ui.label(format!("Confidence: {}", most_active_neuron.1));
		});

	egui::Window::new("Controls")
		.default_size(egui::vec2(150., 0.))
		.show(&ctx, |ui| {
			if ui.button("Reset grid").clicked() {
				for cell in grid.cells.iter_mut().flatten() {
					cell.activation = 0.;
				}
			}

			let slider = egui::Slider::new(loaded_image, 0..=10_000).text("Loaded image");
			if ui.add(slider).changed() {
				if *loaded_image == 0 {
					return;
				}

				let images = network.test_images.clone();
				let labels = network.test_labels.clone();
				let outer_images = images.outer_iter();
				let outer_labels = labels.outer_iter();
				let image = outer_images
					.zip(outer_labels)
					.nth(*loaded_image as usize - 1)
					.unwrap();

				*loaded_label = *image.1.get(0).unwrap();

				for (cell, pixel) in grid.cells.iter_mut().flatten().zip(image.0) {
					cell.activation = *pixel;
				}
			}

			ui.label(format!("Label: {}", &loaded_label));
		});
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
	model.egui.handle_raw_event(event);
}

fn view(app: &App, model: &Model, frame: Frame) {
	let draw = app.draw();
	draw.background().color(BLACK);

	let grid = &model.grid;

	for cell in grid.cells.iter().flatten() {
		cell.draw(&draw);
	}

	draw.ellipse().x_y(0., 0.).radius(5.).color(WHITE);

	draw.to_frame(app, &frame).unwrap();

	model.egui.draw_to_frame(&frame).unwrap();
}
