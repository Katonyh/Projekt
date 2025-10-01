#pragma once 

#include "container/vector.h"
#include "ml/lin_reg/interface.h"

/**Declaration of LinReg, */
namespace ml{
namespace lin_reg{

using namespace container;

class LinReg final : public Interface {

public:
    /*
    creat a model for training data.
    trainInput reference to a const vector with input data (x)
    trainOutput reference to a const vector with output data (y)
    */
    explicit LinReg(const Vector<double>& trainInput, 
        const Vector<double>& trainOutput) noexcept;

    /*virtual destructor that saves over the interface destructor*/
    ~LinReg() noexcept override = default;

    /*make a prediction */
    double predict(double input) const override;

    /*trains the model in a set number of epochs with a learning rate of 1%*/
    bool train(unsigned int epochCount, double learningRate = 0.01);

    LinReg() = delete;                          // no default-constructor
    LinReg(const LinReg&) = delete;             // no copying(constructor)
    LinReg& operator=(const LinReg&) = delete;  // no copying(allocation)
    LinReg(LinReg&&) = delete;                  // no moving the (constructor)    
    LinReg& operator=(LinReg&&) = delete;       // no moving the(allocation)

private:
    /*training data*/
    const Vector<double>& myTrainInput;
    const Vector<double>& myTrainOutput;

    const size_t myTrainSetCount;

    /* Bias = m and weight = k value i y= kx + m */
    double myWeight;
    double myBias;
};
    }// end lin_reg
} //end ml 