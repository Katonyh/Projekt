#include "ml/lin_reg/lin_reg.h"

namespace ml {
namespace lin_reg{
    
namespace {
constexpr size_t min(const size_t x, const size_t y) noexcept
{
    return x <= y ? x : y;
}
} // namespace

LinReg::LinReg(const Vector<double>& trainInput, const Vector<double>& trainOutput) noexcept
    :myTrainInput{trainInput}
    ,myTrainOutput{trainOutput}
    ,myTrainSetCount{min(trainInput.size(), trainOutput.size())}
    ,myWeight{0.5}
    ,myBias{0.5}
{}

double LinReg::predict(double input) const{
    /*y = kx + m*/
    return myWeight * input + myBias;
}

bool LinReg::train(unsigned int epochCount, double learningRate){

    if(myTrainSetCount == 0)    return false;
    if(epochCount == 0)         return false;
    if(learningRate <= 0.0)     return false;
    
    for (unsigned int epoch = 0; epoch < epochCount; epoch++){
        double gradW = 0.0;
        double gradB = 0.0;
        
        for(size_t i{0U}; i < myTrainSetCount; i++){
            const double x   = myTrainInput[i];
            const double y   = myTrainOutput[i];
            const double yref= myWeight * x + myBias;
            const double e   = yref - y;
            
            gradW += e * x;
            gradB += e;
        }
        const double scale = 2.0 / static_cast<double>(myTrainSetCount);
        gradW *= scale;
        gradB *= scale;
        
        myWeight -= learningRate * gradW;
        myBias   -= learningRate * gradB;
    }
    return true;
}
} //end lin_reg 
}//end ml