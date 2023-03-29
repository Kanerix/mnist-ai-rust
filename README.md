# MNIST AI RUST

An AI trained on the mnist dataset. The code is broken, but it does kinda work. In the first step of backpropagation, they AI will greatly reduce the error. But after that, the AI will slowly get worse at guessing. You are tho able to generate images, and some networks like the `network_100.json` shows the patterns it's looking for when generating those images, even tho its a very bad network, with only about 10% accuracy. The best network right now is the `network_low_lr.json` which has about 72% accuraccy on the test dataset.

## Binary usage

You can use the `--help` flag when running the binary to see available features and usage. Make sure to have the `images`, `config`, `data` and `logs` folders, in the same folder as the binary.

### Generate images

- `cargo run -- -m test -i network_low_lr.json --generate-images`

### Controlling dataset iterations

- `cargo run -- -m train -o network.json --iterations=1000`

### Setting learning rate

- `cargo run -- -m train -o network.json -l 0.01`