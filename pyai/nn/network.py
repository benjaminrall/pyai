import pickle
import numpy as np
from pyai.nn.layers.layer import Layer
import pyai.nn.losses as losses
import pyai.nn.optimisers as optimisers
from pyai.backend.progress_bar import ProgressBar

class Network:
    """`Network` groups a linear stack of layers into a sequential neural network model.

    It provides training and inference features on this model.
    """

    def __init__(self, layers: list[Layer] = []) -> None:
        """Creates a `Network` instance.

        Args:
            layers (list[Layer]): Optional list of layers to add to the model.
        """
        self.layers = layers
        self.optimiser = None
        self.loss = None
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
        
    def pop(self) -> None:
        """Removes the last layer in the network.s

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

    def evaluate_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Returns the loss of the network for a given set of inputs and outputs.

        Args:
            x (np.ndarray): The inputs to evaluate the network with.
            y (np.ndarray): The target outputs to evaluate the network with.

        Returns:
            float: The total loss of the network for all outputs.
        """
        return self.loss(self.call(x), y) + self.penalty()

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Returns the accuracy of the network for a given set of inputs and outputs.

        Args:
            x (np.ndarray): Inputs to evaluate the network with.
            y (np.ndarray): Target outputs to evaluate the network with.

        Returns:
            float: The accuracy of the network over all given inputs and outputs.
        """
        result = self.call(x).argmax(axis=1)
        return np.sum(result == y.argmax(axis=1)) / x.shape[0]
        
    def evaluate(self, x: np.ndarray, y: np.ndarray, 
                 batch_size: int = 32, verbose: bool = True) -> tuple[float, float]:
        """Returns the loss and accuracy of the network for a given set of inputs and outputs.

        Args:
            x (np.ndarray): Inputs to evaluate the network with.
            y (np.ndarray): Target outputs to evaluate the network with.
            batch_size (int): The number of samples per batch of computation.
            verbose (bool): Whether to evaluate the network verbosely.

        Returns:
            tuple[float, float]: A tuple of the form (loss, accuracy).
        """
        # Ensures network is compiled before evaluating it
        if not self.compiled:
            raise RuntimeError("Cannot evaluate a network that hasn't been compiled yet.")

        # Creates batch indices iterator using a progress bar if evaluating verbosely
        batch_indices = range(0, x.shape[0], batch_size)
        if verbose: 
            batch_indices = ProgressBar('Evaluating Network', batch_indices, 0.01, 20, False)

        # Calculates the loss and accuracy over all batches
        total_loss, total_accuracy = 0, 0
        for i in batch_indices:
            total_loss += self.evaluate_loss(x[i : i + batch_size], y[i : i + batch_size])
            total_accuracy += self.evaluate_accuracy(x[i : i + batch_size], y[i : i + batch_size])
        loss, accuracy = total_loss / len(batch_indices), total_accuracy / len(batch_indices)

        # Prints the loss and accuracy if evaluating verbosely
        if verbose: 
            print(" - Loss: {:.10f} - Accuracy: {:.2%}".format(loss, accuracy))
        return loss, accuracy
    
    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: bool = True) -> np.ndarray:
        """Generates output predictions for a set of inputs.

        Args:
            x (np.ndarray): Inputs to predict using the network.
            batch_size (int): The number of samples per batch.
            verbose (bool): Whether to predict verbosely.

        Returns:
            np.ndarray: Array containing the network's predictions.
        """
        # Creates batch indices iterator using a progress bar if evaluating verbosely
        batch_indices = range(0, x.shape[0], batch_size)
        if verbose: 
            batch_indices = ProgressBar('Predicting Outputs', batch_indices, 0.01, 20, True)
        
        # Calculates the output for all batches
        y = np.zeros(x.shape[:1] + self.layers[-1].output_shape)
        for i in batch_indices:
            y[i : i + batch_size] = self.call(x[i : i + batch_size])
        return y

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 1, verbose: bool = True, 
            validation_split: float = 0, validation_data: np.ndarray = None, shuffle: bool = True) -> None:
        """Trains the network for a fixed number of epochs.

        Args:
            x (np.ndarray): Inputs to train the network on
            y (np.ndarray): Target outputs to train the network on
            batch_size (int): The number of samples per gradient update.
            epochs (int): Number of epochs to train the network.
            verbose (bool): Whether to train the network verbosely.
            validation_split (float): Fraction of the training data to use as validation data.
            validation_data (np.ndarray): Validation data in the form [inputs, outputs].
            shuffle (bool): Whether to shuffle the training data before each epoch.
        """
        # Compiles the network with default settings if it isn't compiled
        if not self.compiled:
            self.compile()

        # Splits data into training and validation sets
        validation_x = x
        validation_y = y
        if validation_data is not None:
            validation_x = validation_data[0]
            validation_y = validation_data[1]
        elif 0 < validation_split < 1:
            validation_index = x.shape[0] - int(len(x) * validation_split)
            validation_x = x[validation_index:]
            validation_y = y[validation_index:]
            x = x[:validation_index]
            y = y[:validation_index]            

        # Runs through each training epoch
        for epoch in range(epochs):
            # Shuffles the training data before splitting it into batches
            if shuffle:
                p = np.random.permutation(x.shape[0])
                x, y = x[p], y[p]
            
            # Creates an iterator for the batch indices, using a progress bar if training verbosely
            batch_indices = range(0, x.shape[0], batch_size)
            if verbose: batch_indices = ProgressBar(
                'Epoch {:{}d}/{:d}'.format(epoch + 1, len(str(epochs)), epochs), 
                batch_indices, 0.01, 20, False
            )
                
            # Performs the gradient updates for each batch using backpropagation
            for i in batch_indices:
                # Calculates the loss derivatives and averages them over the batch
                derivatives = self.loss.derivative(
                    self.call(x[i : i + batch_size], training=True),
                    y[i : i + batch_size]
                ) / batch_size

                # Performs the backwards pass through the network
                for layer in reversed(self.layers):
                    derivatives = layer.backward(derivatives, self.optimiser)

            # Prints final loss and accuracy measurements for validation data if training verbosely
            if verbose: print(" - Loss: {:.10f} - Accuracy: {:.2%}".format(
                *self.evaluate(validation_x, validation_y, batch_size, False)
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
    
    def save(self, filepath: str) -> None:
        """Serialises and saves a network.

        Args:
            filepath (str): Path to the location at which the network will be saved.
        """
        # Ensures file is saved with the .pyai file extension
        if not filepath.endswith('.pyai'):
            filepath += '.pyai'

        # Saves the network as a binary file
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'Network':
        """Loads and deserialises a network.

        Args:
            filepath (str): Path to the saved network.

        Raises:
            OSError: If the file cannot be loaded correctly.

        Returns:
            Network: The loaded network.
        """        
        # Attempts to open the file and load the network
        try:
            with open(filepath, 'rb') as f:
                network = pickle.load(f)
        except OSError as e:
            raise OSError("Error loading network from file. " + str(e))
        except pickle.UnpicklingError as e:
            raise OSError(
                "Error loading network from file. File specified is not a valid pickled binary file."
            )
        return network
    
    def get_variables(self) -> list[list[np.ndarray]]:
        """Retrieves the variables of all layers in the network.

        Returns:
            list: A list containing the variables of each layer.
        """
        return [layer.get_variables() for layer in self.layers]
    
    def set_variables(self, variables: list[list[np.ndarray]]) -> None:
        """Sets the variables of the network, from a list of layer variables.

        Args:
            variables (list[list[np.ndarray]]): A list containing the variables of each layer.

        Raises:
            ValueError: If the length of the variables doesn't match the amount of layers in the network.
        """
        if len(variables) != len(self.layers):
            raise ValueError("Variables list must contain variables for each layer in the network.")
        for layer, layer_vars in zip(self.layers, variables):
            layer.set_variables(layer_vars)

    def save_variables(self, filepath: str) -> None:
        """Serialises and saves a network's variables.

        Args:
            filepath (str): Path to the location at which the variables will be saved.
        """
        # Ensures file is saved with the .vars file extension
        if not filepath.endswith('.vars'):
            filepath += '.vars'

        # Saves the variables as a binary file
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_variables(), f)

    def load_variables(self, filepath: str) -> None:
        """Loads and deserialises a network's variables.

        Args:
            filepath (str): Path to the saved variables.

        Raises:
            OSError: If the file cannot be loaded correctly.
        """        
        # Attempts to open the file and load the variables
        try:
            with open(filepath, 'rb') as f:
                variables = pickle.load(f)
        except OSError as e:
            raise OSError("Error loading variables from file. " + str(e))
        except pickle.UnpicklingError as e:
            raise OSError(
                "Error loading variables from file. File specified is not a valid pickled binary file."
            )
        
        # Sets the variables for the network
        self.set_variables(variables)
