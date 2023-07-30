from typing import Any
from pyai.layers.layer import Layer
from pyai.backend.progress_bar import ProgressBar
import pyai.losses as losses
import pyai.optimisers as optimisers
import numpy as np

class Network:
    """`Network` groups a linear stack of layers into a neural network model.

    `Network` provides training and inference features on this model.
    """

    def __init__(self, layers: list[Layer] = []) -> None:
        """Creates a `Network` instance.

        Args:
            layers (list[Layer]): Optional list of layers to add to the model.
        """
        self.layers = layers
        self.built = False
        self.compiled = False

    def add(self, layer: Layer) -> None:
        """Adds a layer instance on top of the layer stack.

        Args:
            layer (Layer): An instance of a layer.

        Raises:
            TypeError: If layer is not a layer instance.
        """
        # Ensures the given layer is a layer instance
        if not isinstance(layer, Layer):
            raise TypeError("The added layer must be an instance of the Layer class.")
        
        # Builds new layer if possible
        if self.built:
            layer.build(self.layers[-1].output_shape)

        # Appends the layer to the layer stack
        self.layers.append(layer)
        
    def pop(self) -> Layer:
        """Removes the last layer in the network.

        Raises:
            IndexError: If there are no layers in the model.
        """
        self.layers.pop()

        # Resets built status if the layers stack is now empty
        if len(self.layers) == 0:
            self.built = False
    
    def get_layer(self, index: int) -> Layer:
        """Retrieves a layer based on its index.

        Args:
            index (int): Index of the layer.

        Raises:
            IndexError: If the index is out of range.

        Returns:
            Layer: The layer instance at the given index.
        """
        if index < 0 or index >= len(self.layers):
            raise IndexError("Layer index out of range.")
        return self.layers[index]

    def build(self, input_shape: tuple) -> None:
        """Builds the network based on the given input shape.

        Args:
            input_shape (tuple): The shape of the input to the network.
        """
        # Prevents empty networks from being built.
        if len(self.layers) == 0:
            return
        
        # Passes the input shape through the network, building all layers.
        for layer in self.layers:
            input_shape = layer.build(input_shape)

        self.built = True
   
    def compile(self, optimiser: str | optimisers.Optimiser = 'rmsprop',
                loss: str | losses.Loss = 'categorical_crossentropy') -> None:
        """Configures the network for training.

        Args:
            optimiser (str | Optimiser): Name of an optimiser or an optimiser instance.
            loss (str | Loss): Name of a loss function or a loss function instance.
        """
        self.optimiser = optimisers.get(optimiser)
        self.loss = losses.get(loss)
        self.compiled = True

    def __call__(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        return self.call(inputs, **kwargs)

    def call(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Calls the network on a given set of inputs.

        Args:
            inputs (np.ndarray): An array of inputs with the shape (batches, -1)

        Returns:
            np.ndarray: The outputs of the network.
        """
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs

    def evaluate_loss(self, inputs: np.ndarray, outputs: np.ndarray) -> float:
        """Evaluates the loss of the network for a given set of inputs and outputs."""
        return self.loss(self.call(inputs), outputs) + self.penalty()

    def evaluate_accuracy(self, inputs: np.ndarray, outputs: np.ndarray) -> float:
        """Evaluates the accuracy of the network for a given set of inputs and outputs."""
        result = self.call(inputs).argmax(axis=1)
        return len(np.where(result == outputs.argmax(axis=1))[0]) / inputs.shape[0]
        
    def evaluate(self, inputs: np.ndarray, outputs: np.ndarray) -> list[float]:
        """Returns the loss value and metrics for the model."""
        raise NotImplementedError()

    def fit(self, train_inputs: np.ndarray, train_outputs: np.ndarray, 
            batch_size: int = 1, epochs: int = 1,
            test_inputs: np.ndarray = None, test_outputs: np.ndarray = None,
            verbose: bool = True) -> None:
        """Trains the model on a given set of training inputs and outputs."""
        # Compiles the network with default settings if it isn't compiled
        if not self.compiled:
            self.compile()

        # Determines what data to use for testing the network
        using_test_data = test_inputs is not None and test_outputs is not None
        test_inputs = test_inputs if using_test_data else train_inputs
        test_outputs = test_outputs if using_test_data else train_outputs

        training_samples = train_inputs.shape[0]

        # Trains the network for the given number of epochs
        for epoch in range(epochs):
            # Generates a random permutation of the training inputs and outputs
            p = np.random.permutation(training_samples)
            train_inputs, train_outputs = train_inputs[p], train_outputs[p]

            batch_indices = range(0, training_samples, batch_size)

            # Creates a progress bar if training verbosely        
            if verbose: batch_indices = ProgressBar(
                'Epoch {:{}d}/{:d}'.format(epoch + 1, len(str(epochs)), epochs), 
                batch_indices, 0.01, 20, False
            )

            # Adjusts the network for all training batches using backpropagation
            for i in batch_indices:
                # Calculates loss derivative and averages over the batch
                derivatives = self.loss.derivative(
                    self.call(train_inputs[i : i + batch_size], training=True), 
                    train_outputs[i : i + batch_size]
                ) / batch_size
                
                # Performs the backwards pass
                for layer in reversed(self.layers):
                    derivatives = layer.backward(derivatives, self.optimiser)

            # Prints final loss and accuracy measurements if training verbosely
            if verbose: print(" - Loss: {:.10f} - Accuracy: {:.2%}".format(
                self.evaluate_loss(test_inputs, test_outputs),
                self.evaluate_accuracy(test_inputs, test_outputs)
            ))

    def penalty(self) -> float:
        """Calculates the total regularisation penalty of all layers in the network.

        Returns:
            float: The network's total regularisation penalty.
        """
        return sum([layer.penalty() for layer in self.layers])
    
    def summary(self) -> None:
        """Prints a string summary of the network.

        Raises:
            ValueError: If `summary` is called before the network is built.
        """

        # Ensures that the network is built before printing a summary
        if not self.built:
            raise ValueError(
                "This model has not yet been built. Build the model first " +
                "by calling `build()` or by calling the model on a batch of data."
            )
        
        # Stores all information to be printed in the summary
        types = [type(layer).__name__ for layer in self.layers]
        shapes = [str(layer.output_shape) for layer in self.layers]
        params = [str(layer.parameters) for layer in self.layers]
        
        # Calculates the sizes for each column in the summary table
        cols = [
            max(10, max([len(t) for t in types]) + 5),
            max(10, max([len(s) for s in shapes]) + 12),
            max(10, max([len(p) for p in params]) + 7)
        ]

        # Prints the column headers and divider bars
        print("\n{:=^{}s}".format("", sum(cols) + 8))
        print(" {:<{}s} | {:^{}s} | {:>{}s}".format(
            "Layer", cols[0], "Output Shape", cols[1], "Param #", cols[2]
        ))
        print("{:=^{}s}".format("", sum(cols) + 8))

        # Prints the layer information
        for t, s, p in zip(types, shapes, params):
            print(" {:<{}s} | {:^{}s} | {:>{}s} ".format(t, cols[0], s, cols[1], p, cols[2]))

        # Prints total parameter value
        print("{:=^{}s}".format("", sum(cols) + 8))
        print(f"There are {sum([layer.parameters for layer in self.layers])} total parameters.\n")
   