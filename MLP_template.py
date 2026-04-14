"""
MLP Template for MNIST Classification
Author: Dong Yang
Date: 03-12-2026
https://github.com/DongYang26/CSC4851_6851/tree/main/MNIST

Modified by: Linn Kloefta
Date: 04-14-2026
https://github.com/linnerle/MNISTClassification
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Requirement (4) Sigmoid activation (for the 1st layer).
def sigmoid(x):
    # Calculates 1 / (1 + e^-x) to squash values between 0 and 1
    return 1 / (1 + np.exp(-x))


# Requirement (5) Softmax activation (for the 2nd layer).
def softmax(x):
    # Subtracts max value for numerical stability before exponentiation to avoid overflow
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Divides each row by its sum so all probabilities in a batch item add up to 1.0
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def dataloader(train_dataset, test_dataset, batch_size=128):
    # Requirement (3) batch-based training: Batches data in chunks of 128, shuffling train data
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    # Batches test data identically but without shuffling
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    # Returns the Iterable loader objects
    return train_loader, test_loader


def load_data():
    # Defines standard PyTorch pipeline to turn Image to Tensor and map to [-1, 1] via Normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Fetches MNIST from torchvision, marks it as training, downloads if missing, transforms
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transform)
    # Fetches the validation/testing portion of the MNIST dataset
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform)
    # Logs dataset properties
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    # Wraps the datasets inside Iterable dataloaders
    return dataloader(train_dataset, test_dataset)


# Model Architecture: 2-layer neural network
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):
        # Additional specification (1): Uses Gaussian distribution (randn) to independently initialize weights
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        # Additional specification (1): Initializes biases exactly at zero
        self.b1 = np.zeros((1, hidden_size))
        # Repeats Gaussian distribution for Output Layer weights
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        # Initializes second list of biases at zero
        self.b2 = np.zeros((1, output_size))
        # Stores learning rate internally
        self.lr = lr

    # Requirement (1) Forward propagation
    def forward(self, x):
        # Multiplies batch input by w1 and offsets by b1 to calculate Linear operation Z1
        self.z1 = np.dot(x, self.w1) + self.b1
        # Requirement (1) & (4): Predicts 1st layer Activation A1 by passing Z1 into Sigmoid
        self.a1 = sigmoid(self.z1)
        # Calculates Linear operation Z2 via A1 multiplied by Output weights w2 offset by b2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        # Requirement (1) & (5): Maps Z2 logic via Softmax into probabilities as A2
        self.a2 = softmax(self.z2)
        # Binds Output Layer probablities to explicit 'outputs' reference variable
        outputs = self.a2
        # Requirement (1): Output prediction probabilities after passing through both layers
        return outputs

    # Requirement (2) Backward propagation (Including gradient calculations).
    def backward(self, x, y, pred):
        # Extracts actual batch size count 'm' to properly average out gradients
        m = x.shape[0]

        # Allocates empty zeros matrix matching dimensions of prediction [batch_size, 10]
        y_one_hot = np.zeros_like(pred)
        # Flips exact column indices tied to label integer logic completely to '1.0'
        y_one_hot[np.arange(m), y] = 1

        # Generates Loss Derivative dz2 efficiently because cross-entropy derivative combines cleanly with Softmax
        dz2 = pred - y_one_hot
        # Requirement (2) Computes averaged Gradient for w2 utilizing Chain Rule logic (dz2 mapped back through a1)
        dw2 = np.dot(self.a1.T, dz2) / m
        # Averages scalar gradient for base b2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Projects Output Error backwards through Network & scales against Sigmoid's isolated derivative (a1 * (1-a1))
        dz1 = np.dot(dz2, self.w2.T) * self.a1 * (1 - self.a1)
        # Calculates dw1 matching inputs to dz1 divided by sample batch 'm'
        dw1 = np.dot(x.T, dz1) / m
        # Completes final Chain rule partial derivatives for Base b1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Requirement (2) Updates all parameters via simple vanilla Gradient Descent applying learning rate multiplier
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2

    # Requirement (3) Train function.
    def train(self, x, y):
        # Maps dimension batch sample length dynamically
        m = x.shape[0]

        # Requirement (3) Extrapolates Probabilities logic internally by feeding input to Forward Pipeline
        pred = self.forward(x)

        # Requirement (6) Cross entropy function (For calculating loss)
        # Stores mathematical epsilon threshold logic to prevent np.log computing exactly zero
        epsilon = 1e-15
        # Maps precise predicted probabilities per known labels natively via indexing natively sum mapped to scalar 'Loss'
        loss = -np.sum(np.log(pred[np.arange(m), y] + epsilon)) / m

        # Requirement (3) Finalizes weight state propagation cascading gradients down back
        self.backward(x, y, pred)

        # Requirement (3): Tracks and returns the training loss per step
        return loss


# Evaluation helper function to check accuracy for any given dataloader
def evaluate(model, loader, input_size):
    correct_pred = 0
    total_pred = 0
    for inputs, labels in loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    return correct_pred / total_pred

# Requirement (7) Main function.


def main():
    # Loads Datasets directly from torchvision abstraction wrapper
    train_loader, test_loader = load_data()

    # Pre-defines Image Array 28x28 mapping out to 784 flattened neuron 1D vector parameters
    input_size = 28*28
    # Explicit definition of internal model complexity sizing per task
    hidden_size = 128
    # Predicts integers strictly [0-9] generating 10 possible outputs
    output_size = 10
    # Custom Hyperparameter multiplier manually assigned
    lr = 0.5
    # Requirement (3): Uses Batch Based Training limiting cycles capped exactly iteratively per constraints
    num_epochs = 30

    # Generates standard MLP instance instance passing defined sizes
    model = MLP(input_size, hidden_size, output_size, lr)

    # Tracking metrics for figures
    loss_history = []
    train_acc_history = []
    test_acc_history = []

    # Requirement (3): Iterates across all loaded batch samples
    for epoch in range(num_epochs):
        # Accumulates iteration loss strictly for total averaging per round
        total_loss = 0

        # Requirement (3) Iterate over batches of explicitly provided dataloader samples
        for inputs, labels in train_loader:
            # Requirement (1) Input: Batch of explicitly flattened MNIST images (shape [batch_size, 784])
            x = inputs.view(-1, input_size).numpy()
            # Deconstructs labels strictly from internal backend logic mapping
            y = labels.numpy()
            # Operates direct train logic on variables per round iteratively
            loss = model.train(x, y)
            # Adds extracted sample cost output strictly up towards final reporting
            total_loss += loss

        # Track loss inside history
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Track accuracy for both train and test loaders
        train_acc = evaluate(model, train_loader, input_size)
        test_acc = evaluate(model, test_loader, input_size)
        train_acc_history.append((epoch + 1, train_acc))
        test_acc_history.append((epoch + 1, test_acc))

        # Additional Specification (3): Implements printing training loop logging
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Additional Specification (2): Report/Print Test Accuracy after training mapped
    final_test_acc = evaluate(model, test_loader, input_size)
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    # Return the trained model, metrics, and test loader for reporting purposes
    return model, loss_history, train_acc_history, test_acc_history, test_loader


if __name__ == "__main__":
    # Boots directly via wrapper calling native root script mappings natively
    main()
