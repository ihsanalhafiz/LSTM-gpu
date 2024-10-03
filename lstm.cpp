/*
The basic LSTM model implementation

author: 大火同学
date:   2018/4/28
email:  12623862@qq.com

Modified by: 2024 Sep for GPU acceleration
author: miahafiz
email: 
*/

#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "lstm.h"

using namespace std;

// Activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the activation function
double dsigmoid(double y){
    return y * (1.0 - y);  
}           

// Derivative of tanh
double dtanh(double y){
    y = tanh(y);
    return 1.0 - y * y;  
}

/*
Weight initialization
Parameters:
w、Weight initialization Parameters :w - 2D weight arrayx - number of rows
x、number of rows
y、number of columns
*/
void initW(double **w, int x, int y){
    FOR(i, x){
        FOR(j, y)
            w[i][j] = RANDOM_VALUE();  //Random distribution -1~1
    }
}

/*
Print the status of the lstm cell for debugging
*/
void Lstm::showStates(){
	FOR(s, _states.size()){
		cout<<"states["<<s<<"]:"<<endl<<"I_G\t\tF_G\t\tO_G\t\tN_I\t\tS\t\tH"<<endl;
		FOR(i, _hideNodeNum){
			cout<<_states[s]->I_G[i]<<"\t\t";
			cout<<_states[s]->F_G[i]<<"\t\t";
			cout<<_states[s]->O_G[i]<<"\t\t";
			cout<<_states[s]->N_I[i]<<"\t\t";
			cout<<_states[s]->S[i]<<"\t\t";
			cout<<_states[s]->H[i]<<"\n";
		}
		cout<<"Y:";
		FOR(i, _outNodeNum){
			cout<<_states[s]->Y[i]<<"\t";
		}
		cout<<endl;
	}
}

/*
Clear unit status
*/
void Lstm::resetStates(){
	FOR(i, _states.size()){
		delete _states[i];
	}
	_states.clear();
}

/*
Print weights for debugging
*/
void Lstm::showWeights(){
	cout<<"--------------------Wx+b=Y-----------------"<<endl;
	FOR(i, _outNodeNum){
    	cout<<"_W_Y:\n";
    	FOR(j, _hideNodeNum){
    		cout<<_W_Y[j][i]<<"\t";
    	}
    	cout<<"\n_BY:\n"<<_B_Y[i];
    }

    cout<<"\n\n-------------------------Wx+Uh+b=Y----------------------------"<<endl;
    FOR(j, _hideNodeNum){
    	cout<<"\n------------------\nU_:\n";
    	FOR(k, _hideNodeNum){
    		cout<<_U_I[k][j]<<"|"<<_U_F[k][j]<<"|"<<_U_O[k][j]<<"|"<<_U_G[k][j]<<endl;
    	}
    	cout<<"\nW_:\n";
    	FOR(k, _inNodeNum){
    		cout<<_W_I[k][j]<<"|"<<_W_F[k][j]<<"|"<<_W_O[k][j]<<"|"<<_W_G[k][j]<<endl;
    	}

        cout<<"\nB_:\n";
    	cout<<_B_I[j]<<"|"<<_B_F[j]<<"|"<<_B_O[j]<<"|"<<_B_G[j]<<endl;
    }
    cout<<endl<<"---------------------------------------------------"<<endl;
}

/*
Initialize network weights
*/
void Lstm::renewWeights(){
	initW(_W_I, _inNodeNum, _hideNodeNum);
    initW(_U_I, _hideNodeNum, _hideNodeNum);
    initW(_W_F, _inNodeNum, _hideNodeNum);
    initW(_U_F, _hideNodeNum, _hideNodeNum);
    initW(_W_O, _inNodeNum, _hideNodeNum);
    initW(_U_O, _hideNodeNum, _hideNodeNum);
    initW(_W_G, _inNodeNum, _hideNodeNum);
    initW(_U_G, _hideNodeNum, _hideNodeNum);
    initW(_W_Y, _hideNodeNum, _outNodeNum);

    memset(_B_I, 0, sizeof(double)*_hideNodeNum);
    memset(_B_O, 0, sizeof(double)*_hideNodeNum);
    memset(_B_G, 0, sizeof(double)*_hideNodeNum);
    memset(_B_F, 0, sizeof(double)*_hideNodeNum);
    memset(_B_Y, 0, sizeof(double)*_outNodeNum);
}


/*
Constructor
Parameters:
innode, number of input units (number of features)
hidenode, number of hidden units
outnode, number of output units (result dimension)
*/
Lstm::Lstm(int innode, int hidenode, int outnode){
    _inNodeNum = innode;
    _hideNodeNum = hidenode;
    _outNodeNum = outnode;
	_verification = 0;
	_learningRate = LEARNING_RATE;

    //Dynamically initialize weights
    _W_I = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_F = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_O = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_G = (double**)malloc(sizeof(double*)*_inNodeNum);
    FOR(i, _inNodeNum){
        _W_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _U_I = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_F = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_O = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_G = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _U_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _B_I = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_F = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_O = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_G = (double*)malloc(sizeof(double)*_hideNodeNum);

    _W_Y = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _W_Y[i] = (double*)malloc(sizeof(double)*_outNodeNum);
    }
    _B_Y = (double*)malloc(sizeof(double)*_outNodeNum);

	renewWeights();

    cout<<"Lstm instance inited."<<endl;
}

/*
Destructor, release memory
*/
Lstm::~Lstm(){
	resetStates();

    FOR(i, _inNodeNum){
        if(_W_I[i]!=NULL){
            free(_W_I[i]);
            _W_I[i]=NULL;
        }
        if(_W_F[i]!=NULL){
            free(_W_F[i]);
            _W_F[i]=NULL;
        }
        if(_W_O[i]!=NULL){
            free(_W_O[i]);
            _W_O[i]=NULL;
        }
        if(_W_G[i]!=NULL){
            free(_W_G[i]);
            _W_G[i]=NULL;
        }
    }
    if(_W_I!=NULL){
        free(_W_I);
        _W_I=NULL;
    }
    if(_W_F!=NULL){
        free(_W_F);
        _W_F=NULL;
    }
    if(_W_O!=NULL){
        free(_W_O);
        _W_O=NULL;
    }
    if(_W_G!=NULL){
        free(_W_G);
        _W_G=NULL;
    }

    FOR(i, _hideNodeNum){
        if(_U_I[i]!=NULL){
            free(_U_I[i]);
            _U_I[i]=NULL;
        }
        if(_U_F[i]!=NULL){
            free(_U_F[i]);
            _U_F[i]=NULL;
        }
        if(_U_O[i]!=NULL){
            free(_U_O[i]);
            _U_O[i]=NULL;
        }
        if(_U_G[i]!=NULL){
            free(_U_G[i]);
            _U_G[i]=NULL;
        }
    }
    if(_U_I!=NULL){
        free(_U_I);
        _U_I=NULL;
    }
    if(_U_F!=NULL){
        free(_U_F);
        _U_F=NULL;
    }
    if(_U_O!=NULL){
        free(_U_O);
        _U_O=NULL;
    }
    if(_U_G!=NULL){
        free(_U_G);
        _U_G=NULL;
    }


    if(_B_I!=NULL){
        free(_B_I);
        _B_I=NULL;
    }
    if(_B_F!=NULL){
        free(_B_F);
        _B_F=NULL;
    }
    if(_B_O!=NULL){
        free(_B_O);
        _B_O=NULL;
    }
    if(_B_G!=NULL){
        free(_B_G);
        _B_G=NULL;
    }


    FOR(i, _hideNodeNum){
        if(_W_Y[i]!=NULL){
            free(_W_Y[i]);
            _W_Y[i]=NULL;
        }
    }
    if(_W_Y!=NULL){
        free(_W_Y);
        _W_Y=NULL;
    }
    if(_B_Y!=NULL){
        free(_B_Y);
        _B_Y=NULL;
    }

    cout<<"Lstm instance has been destroyed."<<endl;
}

/*
Calculate the loss of the training set
Parameters:
x, training feature set
y, training label set
*/
double Lstm::trainLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	len -= _verification*len;//Training set length
	FOR(i, len){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
		// delete state;
		// state = NULL;
        _states.push_back(state);
	}
	rmse = error/(len*_outNodeNum);
	return rmse;
}


/*
Calculate the loss of the validation set. The parameters are the same as the previous function. The starting subscript of the validation set is calculated through _verification
Parameters:
x, training feature set
y, training label set
*/
double Lstm::verificationLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	int start = len-_verification*len;//Validation set starting subscript
	if(start==len) return 0;//The number of validation sets is 0
	for(int i=start;i<len;++i){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
        // delete state;
        // state = NULL;
        _states.push_back(state);
	}
	rmse = error/((len-start)*_outNodeNum);
	return rmse;
}


/*
Single sample forward propagation
Parameters:
x, single sample feature vector
*/
LstmStates *Lstm::forward(DataType *x){
	if(x==NULL){
		return 0;
	}

    LstmStates *lstates = new LstmStates(_hideNodeNum, _outNodeNum);
 //    LstmStates *lstates = (LstmStates*)malloc(sizeof(LstmStates));
	// memset(lstates, 0, sizeof(LstmStates));

	if(_states.size()>0){
		memcpy(lstates->PreS, _states[_states.size()-1]->S, sizeof(double)*_hideNodeNum);
		memcpy(lstates->PreH, _states[_states.size()-1]->H, sizeof(double)*_hideNodeNum);
	}

    FOR(j, _hideNodeNum){   
        double inGate = 0.0;
        double outGate = 0.0;
        double forgetGate = 0.0;
        double newIn = 0.0;
        // double s = 0.0;

        FOR(m, _inNodeNum){
            inGate += x[m] * _W_I[m][j]; 
            outGate += x[m] * _W_O[m][j];
            forgetGate += x[m] * _W_F[m][j];
            newIn += x[m] * _W_G[m][j];
        }

        FOR(m, _hideNodeNum){
            inGate += lstates->PreH[m] * _U_I[m][j];
            outGate += lstates->PreH[m] * _U_O[m][j];
            forgetGate += lstates->PreH[m] * _U_F[m][j];
            newIn += lstates->PreH[m] * _U_G[m][j];
        }

        inGate += _B_I[j];
        outGate += _B_O[j];
        forgetGate += _B_F[j];
        newIn += _B_G[j];

        lstates->I_G[j] = sigmoid(inGate);   
        lstates->O_G[j] = sigmoid(outGate);
        lstates->F_G[j] = sigmoid(forgetGate);
        lstates->N_I[j] = tanh(newIn);

        lstates->S[j] = lstates->F_G[j]*lstates->PreS[j]+(lstates->N_I[j]*lstates->I_G[j]);
        // lstates->H[j] = lstates->I_G[j]*tanh(lstates->S[j]);//!!!!!!
        lstates->H[j] = lstates->O_G[j]*tanh(lstates->S[j]);//changed
    }

    double out = 0.0;
    FOR(i, _outNodeNum){
	    FOR(j, _hideNodeNum){
	        out += lstates->H[j] * _W_Y[j][i];
	    }
	    out += _B_Y[i];
	    // lstates->Y[i] = sigmoid(out);
	    lstates->Y[i] = out;
	}

    return lstates;
}

/*
Forward propagation, calculation by batch_size is not yet implemented.
Parameters:
trainSet, training feature set, vector<feature vector (vector length must be the same as the number of input units)>
labelSet, training label set, vector<label vector (vector length must be the same as the number of output units)>
*/
void Lstm::forward(vector<DataType*> trainSet, vector<DataType*> labelSet){
	int len = trainSet.size();
	len -= _verification*len;//减去验证集
	FOR(i, len){
		LstmStates *state = forward(trainSet[i]);
	    //保存标准误差关于输出层的偏导
	    double delta = 0.0;
	    FOR(j, _outNodeNum){//
	    	// delta = (labelSet[i][j]-state->Y[j])*dsigmoid(state->Y[j]);//!!!!!!!!!
	    	// delta = (labelSet[i][j]-state->Y[j]);//changed
	    	delta = 2*(state->Y[j]-labelSet[i][j]);//loss=label^2-2*label*y+y^2;   dloss/dy=2y-2label; 
	    	state->yDelta[j] = delta;
	    }
	    _states.push_back(state);
	}
}

/*
Backward propagation, calculate the partial derivatives of each weight
Parameters:
trainSet, training feature set, vector<feature vector (vector length must be the same as the number of input units)>
deltas, object pointer to store the partial derivatives of each weight
*/
void Lstm::backward(vector<DataType*> trainSet, Deltas *deltas) {
    if (_states.size() <= 0) {
        cout << "need to go forward first." << endl;
        return;
    }

    // Hidden layer deviation, calculated by the hidden layer error at a time point after the current one 
    // and the error of the current output layer
    double hDelta[_hideNodeNum];  // Error in the hidden state (gradient)
    double *oDelta = new double[_hideNodeNum];  // Output gate delta (gradient)
    double *iDelta = new double[_hideNodeNum];  // Input gate delta (gradient)
    double *fDelta = new double[_hideNodeNum];  // Forget gate delta (gradient)
    double *nDelta = new double[_hideNodeNum];  // New input gate delta (gradient)
    double *sDelta = new double[_hideNodeNum];  // State delta (gradient for cell state)

    // Error at the hidden layer of the previous time step
    double *oPreDelta = new double[_hideNodeNum];  // Previous output gate delta
    double *iPreDelta = new double[_hideNodeNum];  // Previous input gate delta
    double *fPreDelta = new double[_hideNodeNum];  // Previous forget gate delta
    double *nPreDelta = new double[_hideNodeNum];  // Previous new input gate delta
    double *sPreDelta = new double[_hideNodeNum];  // Previous state delta
    double *fPreGate = new double[_hideNodeNum];   // Previous forget gate values

    // Initialize all pre-delta values to zero for the start of backpropagation
    memset(oPreDelta, 0, sizeof(double) * _hideNodeNum);
    memset(iPreDelta, 0, sizeof(double) * _hideNodeNum);
    memset(fPreDelta, 0, sizeof(double) * _hideNodeNum);
    memset(nPreDelta, 0, sizeof(double) * _hideNodeNum);
    memset(sPreDelta, 0, sizeof(double) * _hideNodeNum);
    memset(fPreGate, 0, sizeof(double) * _hideNodeNum);

    // Start backpropagation from the last time step and move backward
    int p = _states.size() - 1;
    for (; p >= 0; --p) { // batch=1, process single sample at a time
        // Current hidden layer gate and state values
        double *inGate = _states[p]->I_G;     // Input gate values at time step p
        double *outGate = _states[p]->O_G;    // Output gate values at time step p
        double *forgetGate = _states[p]->F_G; // Forget gate values at time step p
        double *newInGate = _states[p]->N_I;  // New input gate values at time step p
        double *state = _states[p]->S;        // Cell state at time step p
        double *h = _states[p]->H;            // Hidden state output at time step p

        // Hidden layer and cell state values from the previous time step
        double *preH = _states[p]->PreH;      // Hidden state output from previous time step
        double *preState = _states[p]->PreS;  // Cell state from previous time step

        // Update the weights between the hidden layer and the output layer
        FOR(k, _outNodeNum) { // Loop over output layer units
            FOR(j, _hideNodeNum) {  // Loop over hidden layer units
                // Calculate weight update for connection from hidden layer to output layer
                deltas->dwy[j][k].data += _states[p]->yDelta[k] * h[j];
            }
            // Update bias for the output layer
            deltas->dby[k].data += _states[p]->yDelta[k];
        }

        // Calculation of gradients for each hidden unit in the network
        FOR(j, _hideNodeNum) { // Loop over hidden layer units
            // Initialize the deltas (gradients) for this time step
            oDelta[j] = 0.0;
            iDelta[j] = 0.0;
            fDelta[j] = 0.0;
            nDelta[j] = 0.0;
            sDelta[j] = 0.0;
            hDelta[j] = 0.0;

            // Calculate the gradient of the objective function with respect to the hidden state
            FOR(k, _outNodeNum) {
                hDelta[j] += _states[p]->yDelta[k] * _W_Y[j][k]; // Contribution from output layer error
            }
            FOR(k, _hideNodeNum) {
                // Contribution from the next hidden state errors and gate deltas
                hDelta[j] += iPreDelta[k] * _U_I[j][k];
                hDelta[j] += fPreDelta[k] * _U_F[j][k];
                hDelta[j] += oPreDelta[k] * _U_O[j][k];
                hDelta[j] += nPreDelta[k] * _U_G[j][k];
            }

            // Calculate deltas for each gate
            oDelta[j] = hDelta[j] * tanh(state[j]) * dsigmoid(outGate[j]);  // Output gate delta
            sDelta[j] = hDelta[j] * outGate[j] * dtanh(state[j]) + sPreDelta[j] * fPreGate[j]; // State delta
            fDelta[j] = sDelta[j] * preState[j] * dsigmoid(forgetGate[j]);  // Forget gate delta
            iDelta[j] = sDelta[j] * newInGate[j] * dsigmoid(inGate[j]);     // Input gate delta
            nDelta[j] = sDelta[j] * inGate[j] * dtanh(newInGate[j]);        // New input gate delta

            // Update weights between the previous hidden layer and the current hidden layer
            FOR(k, _hideNodeNum) {
                deltas->dui[k][j].data += iDelta[j] * preH[k];  // Update input gate weight
                deltas->duf[k][j].data += fDelta[j] * preH[k];  // Update forget gate weight
                deltas->duo[k][j].data += oDelta[j] * preH[k];  // Update output gate weight
                deltas->dun[k][j].data += nDelta[j] * preH[k];  // Update new input gate weight
            }

            // Update weights between the input layer and the hidden layer
            FOR(k, _inNodeNum) {
                deltas->dwi[k][j].data += iDelta[j] * trainSet[p][k];  // Input gate weight update
                deltas->dwi[k][j].data += fDelta[j] * trainSet[p][k];  // Forget gate weight update
                deltas->dwo[k][j].data += oDelta[j] * trainSet[p][k];  // Output gate weight update
                deltas->dwn[k][j].data += nDelta[j] * trainSet[p][k];  // New input gate weight update
            }

            // Update biases for each gate
            deltas->dbi[j].data += iDelta[j];
            deltas->dbf[j].data += fDelta[j];
            deltas->dbo[j].data += oDelta[j];
            deltas->dbn[j].data += nDelta[j];
        }

        // Free memory if we are at the last time step
        if (p == (_states.size() - 1)) {
            delete[] oPreDelta;
            delete[] fPreDelta;
            delete[] iPreDelta;
            delete[] nPreDelta;
            delete[] sPreDelta;
            delete[] fPreGate;
        }

        // Update previous deltas and gate values for the next iteration
        oPreDelta = oDelta;
        fPreDelta = fDelta;
        iPreDelta = iDelta;
        nPreDelta = nDelta;
        sPreDelta = sDelta;
        fPreGate = forgetGate;
    }

    // Free memory for the final time step deltas
    delete[] oPreDelta;
    delete[] fPreDelta;
    delete[] iPreDelta;
    delete[] nPreDelta;
    delete[] sPreDelta;

    return;
}

/*
Update the weights based on the partial derivatives of the objective function.
Parameters:
deltaSet - Pointer to the object that stores the partial derivatives of each weight.
epoche - The current iteration number (used for adaptive optimizers such as Adam).
*/
void Lstm::optimize(Deltas *deltaSet, int epoche) {
    // Update the weights connecting the hidden layer to the output layer
    FOR(i, _outNodeNum) {  // Loop over each output unit
        FOR(j, _hideNodeNum) {  // Loop over each hidden layer unit
            // Use the optimizer to update the weight between hidden and output layers
            _W_Y[j][i] = deltaSet->dwy[j][i].optimize(_W_Y[j][i], epoche);
        }
        // Update the bias for the output layer
        _B_Y[i] = deltaSet->dby[i].optimize(_B_Y[i], epoche);
    }

    // Update the weights connecting the hidden layers (from previous hidden state to the current hidden state)
    FOR(j, _hideNodeNum) {  // Loop over each hidden layer unit
        FOR(k, _hideNodeNum) {  // Loop over previous hidden layer units
            // Update the weights for input gate (U_I), forget gate (U_F), output gate (U_O), and new input gate (U_G)
            _U_I[k][j] = deltaSet->dui[k][j].optimize(_U_I[k][j], epoche);
            _U_F[k][j] = deltaSet->duf[k][j].optimize(_U_F[k][j], epoche);
            _U_O[k][j] = deltaSet->duo[k][j].optimize(_U_O[k][j], epoche);
            _U_G[k][j] = deltaSet->dun[k][j].optimize(_U_G[k][j], epoche);
        }
        
        // Update the weights connecting the input layer to the hidden layer
        FOR(k, _inNodeNum) {  // Loop over input layer units
            // Update the weights for input gate (W_I), forget gate (W_F), output gate (W_O), and new input gate (W_G)
            _W_I[k][j] = deltaSet->dwi[k][j].optimize(_W_I[k][j], epoche);
            _W_F[k][j] = deltaSet->dwf[k][j].optimize(_W_F[k][j], epoche);
            _W_O[k][j] = deltaSet->dwo[k][j].optimize(_W_O[k][j], epoche);
            _W_G[k][j] = deltaSet->dwn[k][j].optimize(_W_G[k][j], epoche);
        }

        // Update the biases for the input gate, forget gate, output gate, and new input gate in the hidden layer
        _B_I[j] = deltaSet->dbi[j].optimize(_B_I[j], epoche);
        _B_F[j] = deltaSet->dbf[j].optimize(_B_F[j], epoche);
        _B_O[j] = deltaSet->dbo[j].optimize(_B_O[j], epoche);
        _B_G[j] = deltaSet->dbn[j].optimize(_B_G[j], epoche);
    }
}


double _LEARNING_RATE = LEARNING_RATE;


void Lstm::train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold){
	if(trainSet.size()<=0 || labelSet.size()<=0 || trainSet.size()!=labelSet.size()){
		cout<<"data set error!"<<endl;
		return;
	}


    _verification = 0;
    if(verification>0 && verification<0.5){
        _verification = verification;
    }else{
        cout<<"verification rate is invalid."<<endl;
    }

	double lastTrainRmse = 0.0;
	double lastVerRmse = 0.0;
    _LEARNING_RATE = LEARNING_RATE;

    double verificationAvg = 0.0;
    if(_verification>0){
        int verLen = _verification*labelSet.size();
        FOR(i, verLen){
        	verificationAvg += labelSet[labelSet.size()-verLen+i][0];
        }
        verificationAvg /= verLen;
        verificationAvg = verificationAvg<0?-verificationAvg:verificationAvg;
        cout<<"---------------avg="<<verificationAvg<<endl;
    }

    Deltas *deltaSet = new Deltas(_inNodeNum, _hideNodeNum, _outNodeNum);
    cout<<"deltaset inited. start trainning."<<endl;
	FOR(e, epoche){	
		resetStates();
		forward(trainSet, labelSet);
        deltaSet->resetDelta();
		backward(trainSet, deltaSet);
		optimize(deltaSet, e);

		resetStates();
		double trainRmse = trainLoss(trainSet, labelSet);
		double verRmse = verificationLoss(trainSet, labelSet);
		// cout<<"epoche:"<<e<<"|rmse:"<<trainRmse<<endl;
		if(e>0 && abs(trainRmse-lastTrainRmse) < stopThreshold){
			cout<<"train rmse got tiny diff, stop in epoche:"<<e<<endl;
			break;
		}

		if(e>0 && verRmse!=0 && (verRmse-lastVerRmse)>(verificationAvg*0.025)){
			// cout<<"verification rmse ascend too much:"<<verRmse-lastVerRmse<<", stop in epoche:"<<e<<endl;
			// cout<<"verification rmse ascend or got tiny diff, stop in epoche:"<<e<<endl;
			break;
		}

		lastTrainRmse = trainRmse;
		lastVerRmse = verRmse;
	}
    deltaSet->~Deltas();
    deltaSet = NULL;
}

/*
Predict a single sample
Parameters:
x, feature set of the sample to be predicted
*/
DataType *Lstm::predict(DataType *x){
    // cout<<"predict X>"<<endl;
    // FOR(i, _inNodeNum) cout<<x[i]<<",";
    // cout<<endl;

	LstmStates *state = forward(x);
	DataType *ret = new DataType[_outNodeNum];
	memcpy(ret, state->Y, sizeof(DataType)*_outNodeNum);
	// free(state);
	_states.push_back(state);//Remember the unit status at the current time point
    // cout<<"Y>";
    // FOR(i, _outNodeNum) cout<<ret[i]<<",";
    // cout<<endl;
	return ret;
}



//adam optimizer
double Optimizer::adam(double preTheta, const double dt, const int time){
	mt = beta1*mt+(1-beta1)*dt;
	vt = beta2*vt+(1-beta2)*(dt*dt);
	double mcap = mt/(1-pow(beta1, time));
	double vcap = vt/(1-pow(beta2, time));
	double theta = preTheta - (lr*mcap)/(sqrt(vcap)+epsilon);

	// cout<<"Adam-preTheta="<<preTheta<<"|mt="<<mt<<"|vt="<<vt<<"|mcap="<<mcap<<"|vcap="<<vcap<<"|time="<<time<<"|theta="<<theta<<endl;
	return theta;
}

//sgd Optimizer
double Optimizer::sgd(double preTheta, const double dt){
	double theta = preTheta - _LEARNING_RATE*dt;
	return theta;
}

// Initialize the partial derivative set
Deltas::Deltas(const int in, const int hide, const int out){
    _inNodeNum = in;
    _outNodeNum = out;
    _hideNodeNum = hide;

    dwi = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwf = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwo = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwn = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    FOR(i, _inNodeNum){
        dwi[i] = new Delta[_hideNodeNum];
        dwf[i] = new Delta[_hideNodeNum];
        dwo[i] = new Delta[_hideNodeNum];
        dwn[i] = new Delta[_hideNodeNum];
    }

    dui = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duf = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duo = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    dun = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dui[i] = new Delta[_hideNodeNum];
        duf[i] = new Delta[_hideNodeNum];
        duo[i] = new Delta[_hideNodeNum];
        dun[i] = new Delta[_hideNodeNum];
    }

    dbi = new Delta[_hideNodeNum];
    dbf = new Delta[_hideNodeNum];
    dbo = new Delta[_hideNodeNum];
    dbn = new Delta[_hideNodeNum];

    dwy = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dwy[i] = new Delta[_outNodeNum];
    }

    dby = new Delta[_outNodeNum];

}

Deltas::~Deltas(){
    FOR(i, _inNodeNum){
        delete [] dwi[i];
        delete [] dwf[i];
        delete [] dwo[i];
        delete [] dwn[i];
    }
    free(dwi);
    free(dwf);
    free(dwo);
    free(dwn);

    FOR(i, _hideNodeNum){
        delete [] dui[i];
        delete [] duf[i];
        delete [] duo[i];
        delete [] dun[i];
    }
    free(dui);
    free(duf);
    free(duo);
    free(dun);

    FOR(i, _hideNodeNum){
        delete [] dwy[i];
    }
    free(dwy);

    delete [] dbi;
    delete [] dbf;
    delete [] dbo;
    delete [] dbn;
    delete [] dby;
}




LstmStates::LstmStates(const int hide, const int out){
    // std::cout<<"new LstmStates"<<std::endl;
    I_G = (double*)malloc(sizeof(double)*hide);
    F_G = (double*)malloc(sizeof(double)*hide);
    O_G = (double*)malloc(sizeof(double)*hide);
    N_I = (double*)malloc(sizeof(double)*hide);
    S = (double*)malloc(sizeof(double)*hide);
    H = (double*)malloc(sizeof(double)*hide);
    PreS = (double*)malloc(sizeof(double)*hide);
    PreH = (double*)malloc(sizeof(double)*hide);
    Y = (DataType*)malloc(sizeof(DataType)*out);
    yDelta = (double*)malloc(sizeof(double)*out);

    memset(I_G, 0, sizeof(double)*hide);
    memset(F_G, 0, sizeof(double)*hide);
    memset(O_G, 0, sizeof(double)*hide);
    memset(N_I, 0, sizeof(double)*hide);
    memset(S, 0, sizeof(double)*hide);
    memset(H, 0, sizeof(double)*hide);
    memset(PreS, 0, sizeof(double)*hide);
    memset(PreH, 0, sizeof(double)*hide);
    memset(Y, 0, sizeof(DataType)*out);
    memset(yDelta, 0, sizeof(double)*out);
}

LstmStates::~LstmStates(){
    // std::cout<<"delete LstmStates"<<std::endl;
    free(I_G);
    free(F_G);
    free(O_G);
    free(N_I);
    free(S);
    free(H);
    free(PreS);
    free(PreH);
    free(Y);
    free(yDelta);
}


Delta::Delta(){
    opt = new Optimizer();
    data = 0;
}

Delta::~Delta(){
    delete opt;
}

double Delta::optimize(double theta, const int time){
    if(opt!=NULL){
        theta = opt->adam(theta, data, time+1);//time从1开始
        // theta = opt->sgd(theta, data);//time从1开始
    }else{
        theta -= LEARNING_RATE * data;
    }

    return theta;
}

// Reset partial derivatives and save optimizer parameter status
void Deltas::resetDelta(){
    FOR(i, _inNodeNum){
        FOR(j, _hideNodeNum){
            dwi[i][j].data = 0;
            dwf[i][j].data = 0;
            dwo[i][j].data = 0;
            dwn[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _hideNodeNum){
            dui[i][j].data = 0;
            duf[i][j].data = 0;
            duo[i][j].data = 0;
            dun[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _outNodeNum){
            dwy[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        dbi[i].data = 0;
        dbf[i].data = 0;
        dbo[i].data = 0;
        dbn[i].data = 0;
    }

    FOR(i, _outNodeNum){
        dby[i].data = 0;
    }
}


