#include "ml/lin_reg/lin_reg.h"

using container::Vector;
namespace ml{

namespace lin_reg{
    
namespace
{
constexpr size_t min(const size_t x, const size_t y) noexcept
{
    return x <= y ? x : y;
}
} // namespace

LinReg::LinReg(const Vector<double>& trainInput,const Vector<double>& trainOutput)noexcept
    :myTrainInput{trainInput}
    ,myTrainOutput{trainOutput}
    ,myTrainSetCount{min(trainInput.size(), trainOutput.size())}
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
    } //end lin_reg 
}//end ml