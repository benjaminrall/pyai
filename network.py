from layers import Layer
import numpy as np
from losses import Loss

class Network:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers
        self.compiled = False

    def add(self, layer: Layer) -> None:
        self.compiled = False
        self.layers.append(layer)

    def pop(self) -> Layer:
        self.compiled = False
        return self.layers.pop()
    
    def compile(self, input_shape: tuple, loss: str = "mean_squared_err") -> None:
        self.compiled = True
        for layer in self.layers:
            input_shape = layer.build(input_shape)
        self.loss = Loss.get(loss)

    def forward(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def evaluate_cost(self, inputs: np.ndarray, expecteds: np.ndarray) -> float:
        return self.loss.calculate(self.forward(inputs), expecteds)

    def evaluate_accuracy(self, inputs: np.ndarray, expecteds: np.ndarray) -> float:
        outputs = self.forward(inputs).argmax(axis=1)
        return len(np.where(outputs == expecteds.argmax(axis=1))[0]) / inputs.shape[0]
        
    def fit(self, train_inputs: np.ndarray, train_outputs: np.ndarray, 
            test_inputs: np.ndarray, test_outputs: np.ndarray,
            batch_size: int, eta: float, epochs: int, 
            verbose: bool = True) -> None:
        if not self.compiled:
            self.compile(train_inputs.shape[1:])

        for epoch in range(epochs):
            p = np.random.permutation(train_inputs.shape[0])
            train_inputs, train_outputs = train_inputs[p], train_outputs[p]
            for i in range(0, train_inputs.shape[0], batch_size):
                derivatives = self.loss.derivative(
                    self.forward(train_inputs[i : i + batch_size]), train_outputs[i : i + batch_size]
                )
                for layer in reversed(self.layers):
                    derivatives = layer.backward(derivatives, eta)
            if verbose: 
                print(f"Epoch {epoch + 1} complete. | Cost: {self.evaluate_cost(test_inputs, test_outputs)} | Accuracy: {self.evaluate_accuracy(test_inputs, test_outputs) * 100}%")


    def summary(self) -> None:
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