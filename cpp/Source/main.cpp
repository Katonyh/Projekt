/**
 * @brief Linear regression demonstration.
 */
#include <iostream>
#include <vector>

#include "ml/lin_reg/lin_reg.h"

namespace
{
/**
 * @brief Predict with the given linear regression model.
 * 
 * @param[in] linReg Linear regression model to predict with.
 * @param[in] inputData Input data to predict with.
 * @param[in] ostream Output stream to use (default = terminal print).
 */
void predict(const ml::lin_reg::Interface& linReg, const std::vector<double>& inputData,
             std::ostream& ostream = std::cout) noexcept
{
    // Terminate the function if no input data is provided.
    if (inputData.empty())
    {
        ostream << "No input data!\n";
        return;
    }
    ostream << "--------------------------------------------------------------------------------\n";

    // Perform prediction with each input value, print the result in the terminal.
    for (const auto& input : inputData)
    {
        const auto prediction{linReg.predict(input)};
        ostream << "Input: " << input << ", predicted output: " << prediction << "\n";
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}
} // namespace

/**
 * @brief Train and predict with a linear regression model.
 * 
 * @return 0 on success, or -1 on failure.
 */
int main()
{
    // Implement the number of epochs to train and the learning rate as compile-time constants.
    constexpr std::size_t epochCount{1000};
    constexpr double learningRate{0.1};

    // Create training data vectors.
    const std::vector<double> trainInput{0, 1, 2, 3, 4};
    const std::vector<double> trainOutput{2, 4, 6, 8, 10};

    // Create the linear regression model.
    ml::lin_reg::LinReg linReg{trainInput, trainOutput};

    // Train the model, terminate the program with error code -1 on failure.
    if (!linReg.train(epochCount, learningRate))
    {
        std::cout << "Training failed!\n";
        return -1;
    }
    
    // Perform prediction with the model, then terminate the program.
    predict(linReg, trainInput);
    return 0;
}