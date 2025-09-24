#pragma once

namespace ml
{
namespace lin_reg
{
class Interface
{
public:
    /* @brief Delete the linear regression algorithm.*/
    virtual ~Interface() noexcept = default;

    /*@brief Predict based on the given input.
    @param[in] input The input for which to predict.
    @return The predicted value.*/
    virtual double predict(double input) const = 0;
};
} // namespace lin_reg
} //namespace ml