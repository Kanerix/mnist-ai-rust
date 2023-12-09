use clap::{Parser, ValueEnum};
use mnist_ai_rust::network::Network;

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
	#[arg(short, long, default_value_t = false)]
	generate_images: bool,
}

#[tokio::main]
async fn main() {
	let args = Args::parse();

	log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

	let mut network = Network::new(args.learning_rate, (784, 16, 16, 10));

	if let Some(file) = args.input {
		network.load_layers(file).unwrap();
	}

	match args.mode {
		Mode::Train => network.train(args.iterations + 1),
		Mode::Test => network.test(),
	}

	if args.generate_images {
		network.generate_images();
	}

	if let Some(file) = args.output {
		network.save_layers(file).unwrap();
	}
}
