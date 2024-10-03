#include "dataproc.h"

using namespace std;


// Scaling: This function scales the input data by dividing each element by the maximum absolute value.
double scale(double *data, const int len){
    double max = 0;
    FOR(i, len){
        double tmp = data[i] > 0 ? data[i] : -data[i];  // Get the absolute value of each element
        if(max < tmp){
            max = tmp;  // Update the max value if the current value is larger
        }
    }

    double scaleRate = max;  // Store the scaling factor (maximum absolute value)
    FOR(i, len){
        data[i] /= scaleRate;  // Scale each element by the maximum absolute value
    }

    return scaleRate;  // Return the scaling factor
}

// Inverse scaling: This function restores the original scale of the data using the scaling factor.
void invertScale(double *data, const int len, double scaleRate){
    FOR(i, len){
        data[i] *= scaleRate;  // Multiply each element by the scaling factor to restore the original values
    }
}
