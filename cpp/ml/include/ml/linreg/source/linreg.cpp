#include "ml/lin_reg/lin_reg.h"
#include <algorithm>

namespace ml::lin_reg{

LinReg::LinReg(const std::vector<double>& trainInput,const std::vector<double>& trainOutput)noexcept

    :myTrainInput{trainInput}
    ,myTrainOutput{trainOutput}
    ,myTrainSetCount{std::min(trainInput.size(), trainOutput.size())}
    ,myWeight{0.5}
    ,myBias{0.5}
    

{}
double LinReg::predict(double input)const{
    /*y = kx + m*/
    return myWeight * input + myBias;
}

bool LinReg::train(unsigned int epochCount, double learningRate){

    if(myTrainSetCount == 0)    return false;
    if(epochCount == 0)         return false;
    if(learningRate <= 0.0)     return false;

    return true;
}

}//end namespace