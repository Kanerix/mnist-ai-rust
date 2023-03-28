mod layers;
mod network;
mod neuron;
mod utils;

use clap::{Parser, ValueEnum};
use network::Network;

#[derive(ValueEnum, Clone, Debug)]
enum Mode {
	Train,
	Test,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	#[arg(short, long)]
	mode: Mode,
	#[arg(long, default_value_t = 100)]
	iterations: usize,
	#[arg(short, long, default_value_t = 0.1)]
	learning_rate: f32,
	#[arg(short, long, default_value = None)]
	input: Option<String>,
	#[arg(short, long, default_value = None)]
	output: Option<String>,
}

fn main() {
	let args = Args::parse();

    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

	let mut network = Network::new(0.1, (784, 16, 16, 10));

	match args.input {
		None => {},
		Some(file) => {
			network.load_layers(file).unwrap();
		}
	}

	match args.mode {
		Mode::Train => {
			network.train(args.iterations);
		}
		Mode::Test => {
			network.test();
		}
	}

	match args.output {
		None => {},
		Some(file) => {
			network.save_layers(file).unwrap();
		}
	}
}
