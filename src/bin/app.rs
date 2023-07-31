use clap::{arg, command, Parser};
use mnist_ai_rust::network::Network;
use winit::{
	event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	#[arg(short, long)]
	input: String,
}
fn main() {
	let args = Args::parse();

	env_logger::init();

	let mut network = Network::new(0.1, (784, 16, 16, 10));
	network.load_layers(args.input).unwrap();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	event_loop.run(move |event, _, control_flow| match event {
		Event::WindowEvent {
			ref event,
			window_id,
		} if window_id == window.id() => match event {
			WindowEvent::CloseRequested
			| WindowEvent::KeyboardInput {
				input:
					KeyboardInput {
						state: ElementState::Pressed,
						virtual_keycode: Some(VirtualKeyCode::Escape),
						..
					},
				..
			} => *control_flow = ControlFlow::Exit,
			_ => {}
		},
		_ => {}
	});
}
