Training Process

The neural network was trained on a generated dataset with a linear relationship (
). Below are the details of the training:

Input Features: 1 feature.

Target Values: Continuous target values with Gaussian noise.

Training Configuration:

1 input neuron.

10 neurons in the hidden layer with ReLU activation.

1 output neuron with linear activation.

Learning rate: 0.01.

Epochs: 1000.

During training, the cost (Mean Squared Error) was calculated and printed every 100 epochs. The decreasing trend in cost demonstrates the learning progress of the model:

Sample Cost Output:
![image](https://github.com/user-attachments/assets/141f9a8f-5ddb-419a-9804-c470f20c6a6b)
Final Results

After training, the model's performance was evaluated using the Mean Squared Error (MSE) on the dataset:

Final Mean Squared Error: 0.0002

The low MSE indicates that the neural network effectively captured the linear relationship between the input and target values, even with added noise.

Prediction Analysis

The model's predictions closely matched the true target values, demonstrating its ability to generalize the learned relationship.


