#pragma once 

#include "container/vector.h"
#include "ml/lin_reg/interface.h"

/**Deklaration av LinReg, en enkel linj�r regressionsmodell */
namespace ml{
namespace lin_reg{

using namespace container;

class LinReg final : public Interface {

public:
    /*
    skapar en modell med tr�ningsdata
    trainInput referens till en const vektor med idata (x)
    trainOutput referens till en const vektor med utdata (y)
    */
    explicit LinReg(const Vector<double>& trainInput, 
        const Vector<double>& trainOutput) noexcept;

    /*virtuell destruktor som �verlagrar interfacets destruktor*/
    ~LinReg() noexcept override = default;

    /*G�r en prediktion */
    double predict(double input) const override;

    /*tr�nar modellen i ett antal epoker och ger en l�rhastighet p� 1%*/
    bool train(unsigned int epochCount, double learningRate = 0.01);

    LinReg() = delete;                          // ingen default-konstruktor
    LinReg(const LinReg&) = delete;             // ingen kopiering (konstruktor)
    LinReg& operator=(const LinReg&) = delete;  // ingen kopiering (tilldelning)
    LinReg(LinReg&&) = delete;                  // ingen flytt (konstrukto)    
    LinReg& operator=(LinReg&&) = delete;       // ingen flytt (tilldelning)

private:
    /*tr�ningsdata*/
    const Vector<double>& myTrainInput;
    const Vector<double>& myTrainOutput;

    const size_t myTrainSetCount;

    /* Bias = m och weight = k v�rdet i y= kx + m */
    double myWeight;
    double myBias;
};
    }// end lin_reg
} //end ml 