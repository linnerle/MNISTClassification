# MNIST Classification

CSC4851 Spring 2026

This repository contains the implementation of machine learning models for classifying handwritten digits from the MNIST dataset. It features both a completely from-scratch Multi-Layer Perceptron (MLP) built using only NumPy, and a Convolutional Neural Network (CNN) built using PyTorch. For a detailed analysis of the models' performance and evaluation, please refer to the project report located in the `report/` directory.

## Setup

This project was developed using Python 3.13.7 and assumes you have it installed on your system.

1. Clone the project

    ```bash
    git clone https://github.com/linnerle/MNISTClassification.git # clones project into current dir
    cd MNISTClassification/ # navigates into the project folder
    ```

2. Create venv and install dependencies

    ```bash
    python3 -m venv venv # creates venv
    source venv/bin/activate # activates venv
    pip install -r requirements.txt # installs dependencies
    ```

3. Run the project

    ```bash
    python3 MLP_template.py
    python3 CNN_template.py
    ```

    Terminal output prints the training loop with logging; example from `MLP_template.py`:

    ```bash
    The number of training data: 60000
    The number of testing data: 10000
    Epoch 1/30, Loss: 0.6060438884279873
    ...
    Epoch 30/30, Loss: 0.01964185947754861
    Test Accuracy: 0.9787
    ```
