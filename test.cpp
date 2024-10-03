/*
This case is only to demonstrate the usage of this LSTM. The data has not been preprocessed, and the learning result is not very good.
LSTMs can retain the state from previous time steps, making them suitable for processing time-series data.
However, in this case, the dataset is completely random, so there is no temporal correlation.

This example uses an LSTM network to learn the function z = x^2 - xy + y^2. 
x and y are used as training features, and the calculated z is used as training labels. 
The `train` function is called to perform training.
After training, the `predict` function is used to test the learning outcome.

author: Dahuo
date: 2018/5/15
email: 12623862@qq.com
*/

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include "lstm.h"

using namespace std;

#define INPUT  2  // Number of input nodes (x, y)
#define HIDE   20  // Number of hidden nodes in the LSTM layer
#define OUTPUT 1  // Number of output nodes (z)

/*
Function: z = x^2 - xy + y^2
This function calculates the target value (z) based on inputs x and y.
*/
double test_function(double x, double y) {
    double z = x * x - x * y + y * y;
    return z;
}

int main() {
    vector<double *> trainSet;  // Vector to store the training input data
    vector<double *> labelSet;  // Vector to store the corresponding labels

    // Set a fixed random seed to ensure consistent results across runs
    unsigned long seed = 12345678;
    srand(seed);

    // Randomly generate 10,000 sets of training data
    FOR(i, 10000) {
        double *input = (double*)malloc(sizeof(double) * INPUT);  // Allocate memory for input features (x, y)
        double *label = (double*)malloc(sizeof(double) * OUTPUT);  // Allocate memory for the output (z)
        double x = RANDOM_VALUE();  // Generate a random value for x
        double y = RANDOM_VALUE();  // Generate a random value for y
        double z = test_function(x, y);  // Calculate z based on x and y
        input[0] = x;
        input[1] = y;
        label[0] = z;
        trainSet.push_back(input);  // Store input in the training set
        labelSet.push_back(label);  // Store label in the label set
    }

    // Initialize the LSTM network with 2 input nodes, 1000 hidden nodes, and 1 output node
    Lstm *lstm = new Lstm(INPUT, HIDE, OUTPUT);

    // Begin training the LSTM on the dataset
    cout << "/*** Learning function: z = x^2 - xy + y^2 ***/" << endl;
    lstm->train(trainSet, labelSet, 100000, 0.1, 0.00000001);  // Train for 1,000,000 epochs with a threshold of 0.000001

    // Randomly generate 10 test samples and compare the true function result with the LSTM's prediction
    FOR(i, 10) {
        double *test = (double*)malloc(sizeof(double) * INPUT);  // Allocate memory for test input (x, y)
        test[0] = RANDOM_VALUE();  // Random x value
        test[1] = RANDOM_VALUE();  // Random y value
        double *z = lstm->predict(test);  // Predict the output z using the trained LSTM
        double rz = test_function(test[0], test[1]);  // Calculate the real z using the test_function
        double diff = abs(z[0] - rz);  // Calculate the deviation between predicted and real z
        cout << "test " << i << " x=" << test[0] << ", y=" << test[1] << ", predict z=" << z[0] 
             << ", real z=" << rz << ", deviation = " << diff << endl;
        free(test);  // Free memory for test input
        free(z);  // Free memory for predicted output
    }

    // Clean up the LSTM object
    lstm->~Lstm();

    // Free the allocated memory for training inputs and labels
    FOR(i, trainSet.size()) {
        free(trainSet[i]);
        free(labelSet[i]);
    }
    trainSet.clear();  // Clear the training set
    labelSet.clear();  // Clear the label set

    return 0;
}
