from pyai.layers.layer import Layer
from pyai.backend.progress import ProgressBar
import pyai.losses as losses
import pyai.optimisers as optimisers
import numpy as np

class Network:
    """A sequential neural network model."""

    def __init__(self, layers: list[Layer] = []) -> None:
        """Creates an instance of the network with an optional list of layers."""
        self.layers = layers
        self.compiled = False

    def add(self, layer: Layer) -> None:
        """Adds a layer to the network."""
        self.layers.append(layer)

    def pop(self) -> Layer:
        """Removes the last layer of the network."""
        return self.layers.pop()
    
    def compile(self, loss: str | losses.Loss = "mean_squared_error",
                optimiser: str | optimisers.Optimiser = 'sgd') -> None:
        """Configures the network for training."""
        self.loss = losses.get(loss)
        self.optimiser = optimisers.get(optimiser)
        self.compiled = True

    def build(self, input_shape: tuple) -> None:
        """Builds the network based on the given input shape."""
        for layer in self.layers:
            input_shape = layer.build(input_shape)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Feeds an input forward through all layers in the network."""
        for layer in self.layers:
            input = layer(input)
        return input
    
    def variables(self) -> list[np.ndarray]:
        """Returns a list containing all of the variables in the network."""
        return [v for layer in self.layers for v in layer.variables()]
    
    def gradients(self) -> list[np.ndarray]:
        """Returns a list containing all of the variables' gradients in the network."""
        return [g for layer in self.layers for g in layer.gradients()]
    
    def evaluate_loss(self, inputs: np.ndarray, outputs: np.ndarray) -> float:
        """Evaluates the loss of the network for a given set of inputs and outputs."""
        return self.loss(self.forward(inputs), outputs) + self.penalty()
    
    def penalty(self) -> float:
        """Returns the total regularisation penalty of all layers in the network."""
        return sum([layer.penalty() for layer in self.layers])

    def evaluate_accuracy(self, inputs: np.ndarray, outputs: np.ndarray) -> float:
        """Evaluates the accuracy of the network for a given set of inputs and outputs."""
        result = self.forward(inputs).argmax(axis=1)
        return len(np.where(result == outputs.argmax(axis=1))[0]) / inputs.shape[0]
        
    def evaluate(self, inputs: np.ndarray, outputs: np.ndarray) -> list[float]:
        """Returns the loss value and metrics for the model."""
        raise NotImplementedError()

    def fit(self, train_inputs: np.ndarray, train_outputs: np.ndarray, 
            batch_size: int = 1, epochs: int = 1,
            test_inputs: np.ndarray = None, test_outputs: np.ndarray = None,
            verbose: bool = True) -> None:
        """Trains the model on a given set of training inputs and outputs."""
        if not self.compiled:
            self.compile()

        # Determines what data to use for testing the network
        using_test_data = test_inputs is not None and test_outputs is not None
        test_inputs = test_inputs if using_test_data else train_inputs
        test_outputs = test_outputs if using_test_data else train_outputs

        # Trains the network for the given number of epochs
        for epoch in range(epochs):
            # Generates a random permutation of the training inputs and outputs
            p = np.random.permutation(train_inputs.shape[0])
            train_inputs, train_outputs = train_inputs[p], train_outputs[p]

            batch_indices = range(0, train_inputs.shape[0], batch_size)

            # Creates a progress bar if training verbosely        
            if verbose: batch_indices = ProgressBar(
                'Epoch {:{}d}/{:d}'.format(epoch + 1, len(str(epochs)), epochs), 
                batch_indices, 0.001, 20, False
            )

            # Adjusts the network for all training batches using backpropagation
            for i in batch_indices:
                derivatives = self.loss.derivative(
                    self.forward(train_inputs[i : i + batch_size]), 
                    train_outputs[i : i + batch_size]
                )

                # Performs the backwards pass
                for layer in reversed(self.layers):
                    derivatives = layer.backward(derivatives)

                # Adjusts network variables using the optimiser
                self.optimiser(self.variables(), self.gradients())

            # Prints final loss and accuracy measurements if training verbosely
            if verbose: print(" - Loss: {:.10f} - Accuracy: {:.2%}".format(
                self.evaluate_loss(test_inputs, test_outputs),
                self.evaluate_accuracy(test_inputs, test_outputs)
            ))

    def summary(self) -> None:
        """Prints a summary of the network's layers and parameters."""
        if not self.compiled:
            print("Network has not yet been compiled.")
            return
        print(f"\nNetwork has been successfully compiled for the shape: {self.layers[0].input_shape}")
        print("\n {:<12s} | {:^20s} | {:>10s}".format("Layer", "Output Shape", "Param #"))
        print("{:-^50s}".format(""))
        for layer in self.layers:
            print(" {:<12s} | {:^20s} | {:>10d}".format(
                type(layer).__name__, str(layer.output_shape), layer.parameters
            ))
        print(f"\nThere are {sum([layer.parameters for layer in self.layers])} total parameters.")