# Simple AI - Technical Exercise  

This project is a basic neural network implemented from scratch in Python. It is designed as a technical exercise to understand forward propagation, backward propagation, and weight updates using gradient descent.

## Installation  

No external dependencies are required. Ensure you have Python 3 installed.  

## Usage  

Run the script using:  

```bash
python simple_ai.py

python3 simple_ai.py


### Debugging in VS Code

To debug in VS Code, configure the launch.json file in the .vscode directory.
Ensure the correct Python interpreter is set in your environment.

### Overview
This script implements a simple neural network with:

Forward propagation: Computing activations from inputs to output.

Backward propagation: Adjusting weights based on errors.

Gradient descent: Optimizing weights using calculated gradients.

### Key Components
Weights (W1, W2): Trainable parameters of the model.

Biases (b1, b2): Adjustments added to each neuron.

Activation functions: Uses the sigmoid function for non-linearity.

Loss calculation: Measures the difference between predictions and actual values.

### Notes
This is purely an educational exercise and is not optimized for production use.

### License
This project is open-source and free to use for learning purposes.
