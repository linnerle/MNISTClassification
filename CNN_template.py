"""
CNN Template for MNIST Classification
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

# ===================== Utility Functions ===================== #


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #


class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size
        self.lr = lr

        # Convolutional layer (1 filter)
        self.kernel = np.random.randn(
            kernel_size, kernel_size) * 0.05

        # Fully connected layer
        self.conv_out_size = input_size - kernel_size + 1
        self.flattened_size = self.conv_out_size * self.conv_out_size

        self.fc_weights = np.random.randn(
            self.flattened_size, fc_output_size) * 0.05
        self.fc_biases = np.zeros(fc_output_size)

    def forward(self, x):
        """ Forward propagation """
        batch_size = x.shape[0]
        # x may be [batch_size, 1, 28, 28] or [batch_size, 28, 28]. Normalize to [batch_size, 28, 28]
        if len(x.shape) == 4:
            x = x.squeeze(1)

        self.x = x  # Save for backprop
        self.conv_out = np.zeros(
            (batch_size, self.conv_out_size, self.conv_out_size))

        for i in range(self.conv_out_size):
            for j in range(self.conv_out_size):
                region = x[:, i:i+self.kernel_size, j:j+self.kernel_size]
                self.conv_out[:, i, j] = np.sum(
                    region * self.kernel, axis=(1, 2))

        self.relu_out = relu(self.conv_out)
        self.flattened = self.relu_out.reshape(batch_size, -1)

        self.fc_out = np.dot(self.flattened, self.fc_weights) + self.fc_biases
        outputs = softmax(self.fc_out)
        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        batch_size = x.shape[0]

        # 1. one-hot encode the labels
        one_hot_y = np.zeros_like(pred)
        one_hot_y[np.arange(batch_size), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        d_out = pred - one_hot_y  # [batch_size, num_classes]

        # 3. Calculate fully connected layer gradient
        d_fc_weights = np.dot(self.flattened.T, d_out) / batch_size
        d_fc_biases = np.sum(d_out, axis=0) / batch_size
        # [batch_size, flattened_size]
        d_flattened = np.dot(d_out, self.fc_weights.T)

        # 4. Backpropagate through ReLU
        d_relu_out = d_flattened.reshape(
            batch_size, self.conv_out_size, self.conv_out_size)
        d_conv_out = d_relu_out * (self.conv_out > 0)

        # 5. Calculate convolution kernel gradient
        d_kernel = np.zeros_like(self.kernel)
        for i in range(self.conv_out_size):
            for j in range(self.conv_out_size):
                region = self.x[:, i:i+self.kernel_size, j:j+self.kernel_size]
                d_kernel += np.sum(region *
                                   d_conv_out[:, i, j, np.newaxis, np.newaxis], axis=0)
        d_kernel /= batch_size

        # 6. Update parameters
        self.fc_weights -= self.lr * d_fc_weights
        self.fc_biases -= self.lr * d_fc_biases
        self.kernel -= self.lr * d_kernel

    def train(self, x, y):
        # call forward function
        pred = self.forward(x)

        # calculate loss
        batch_size = x.shape[0]
        loss = - \
            np.sum(np.log(pred[np.arange(batch_size), y] + 1e-8)) / batch_size

        # call backward function
        self.backward(x, y, pred)

        return loss

# ===================== Training Process ===================== #


def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 5
    num_filters = 1
    kernel_size = 7
    fc_output_size = 10
    lr = 0.05

    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr)

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.numpy()
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss

        # print the loss for each epoch
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.numpy()
        y = labels.numpy()
        # the model refers to the model that was trained during the raining phase
        pred = model.forward(x)
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")


if __name__ == "__main__":
    main()
