#ifndef __H_LSTM_H__
#define __H_LSTM_H__

using namespace std;

#define LEARNING_RATE  0.0001
#define RANDOM_VALUE() ((double)rand()/RAND_MAX*2-1)   // Generates a random value between -1 and 1
#define FOR(i,N) for(int i=0;i<N;++i)  // Loop macro for compact syntax

typedef double DataType;

class LstmStates{

public:
    double *I_G;       // Input gate
    double *F_G;       // Forget gate
    double *O_G;       // Output gate
    double *N_I;       // New input
    double *S;         // Cell state (memory cell)
    double *H;         // Hidden layer output
    DataType *Y;       // Output values
    double *yDelta;    // Gradient of the error with respect to the output

    double *PreS;      // Previous time step's cell state
    double *PreH;      // Previous time step's hidden layer output

    LstmStates(const int hide, const int out);
    ~LstmStates();
};

class Optimizer{
private:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double mt;
    double vt;
public:
    Optimizer(){
        // Adam optimizer parameters
        lr = 0.01;            // Learning rate
        beta1 = 0.9;          // First moment estimate decay rate
        beta2 = 0.99;         // Second moment estimate decay rate
        epsilon = 1e-8;       // Small constant to prevent division by zero
        mt = 0.0;             // First moment estimate (initialized to 0)
        vt = 0.0;             // Second moment estimate (initialized to 0)
    };
    ~Optimizer(){};
    double adam(double theta, const double dt, const int time); // Adam optimization algorithm
    double sgd(double theta, const double dt); // Stochastic Gradient Descent (SGD) optimization algorithm
};

class Delta{
    Optimizer *opt;
public:
    double data;
    Delta();
    ~Delta();
    double optimize(double theta, const int time); // Applies optimization method (e.g., Adam, SGD)
};

class Deltas{
private:
    int _inNodeNum,_hideNodeNum,_outNodeNum; // Number of nodes for input, hidden, and output layers
public:
    // Gradient matrices, storing gradients for each weight to update them
    Delta **dwi;  // Gradient for weights from input to input gate
    Delta **dui;  // Gradient for weights from hidden state to input gate
    Delta *dbi;   // Gradient for bias in input gate
    Delta **dwf;  // Gradient for weights from input to forget gate
    Delta **duf;  // Gradient for weights from hidden state to forget gate
    Delta *dbf;   // Gradient for bias in forget gate
    Delta **dwo;  // Gradient for weights from input to output gate
    Delta **duo;  // Gradient for weights from hidden state to output gate
    Delta *dbo;   // Gradient for bias in output gate
    Delta **dwn;  // Gradient for weights from input to new input (candidate)
    Delta **dun;  // Gradient for weights from hidden state to new input (candidate)
    Delta *dbn;   // Gradient for bias in new input (candidate)
    Delta **dwy;  // Gradient for weights from hidden layer to output layer
    Delta *dby;   // Gradient for bias in output layer

    Deltas(int in, int hide, int out);
    ~Deltas();
    void resetDelta(); // Reset all gradients to zero
};

class Lstm{
public:
    Lstm(int innode, int hidenode, int outnode);  // Constructor initializing LSTM with specified number of nodes
    ~Lstm();
    void train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold); // Training process
    DataType *predict(DataType *X); // Predicts output based on input X
    void showStates(); // Displays the states (cell state, hidden state) for inspection
    void showWeights(); // Displays the weights for inspection

private:
    LstmStates *forward(DataType *x); // Forward propagation for a single sample
    void forward(vector<DataType*> trainSet, vector<DataType*> labelSet); // Forward propagation for all samples
    void backward(vector<DataType*> trainSet, Deltas *deltaSet); // Backpropagation to update gradients
    void optimize(Deltas *deltaSet, int epoche); // Update the weights using the calculated gradients
    double trainLoss(vector<DataType*> x, vector<DataType*> y); // Calculate training loss (using RMSE - root mean square error)
    double verificationLoss(vector<DataType*> x, vector<DataType*> y); // Calculate verification loss
    void resetStates(); // Reset the states of LSTM units
    void renewWeights(); // Update and renew weights

    int _inNodeNum;  // Number of input nodes
    int _hideNodeNum; // Number of hidden layer nodes
    int _outNodeNum;  // Number of output nodes
    float _verification;  // Proportion of dataset used for verification
    vector<LstmStates*> _states;  // States of hidden units (cell state, hidden state)
    double _learningRate;  // Learning rate

    double **_W_I;    // Weight matrix from input to input gate
    double **_U_I;    // Weight matrix from previous hidden state to input gate
    double *_B_I;     // Bias for input gate
    double **_W_F;    // Weight matrix from input to forget gate
    double **_U_F;    // Weight matrix from previous hidden state to forget gate
    double *_B_F;     // Bias for forget gate
    double **_W_O;    // Weight matrix from input to output gate
    double **_U_O;    // Weight matrix from previous hidden state to output gate
    double *_B_O;     // Bias for output gate
    double **_W_G;    // Weight matrix from input to candidate (new input)
    double **_U_G;    // Weight matrix from previous hidden state to candidate (new input)
    double *_B_G;     // Bias for candidate (new input)
    double **_W_Y;    // Weight matrix from hidden layer to output layer
    double *_B_Y;     // Bias for output layer
};

#endif //__H_LSTM_H__
