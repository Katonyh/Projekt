/**
 * @brief Demonstration of GPIO device drivers in C++:
 * 
 *        The following devices are used:
 *            - A button connected to pin 13 on the device toggles a timer.
 *            - The aforementioned timer toggles an LED every 100 ms when enabled.
 *            - Another timer reduces the effect of contact bounces after pushing the button.
 *            - A watchdog timer is used to restart the program if it gets stuck somewhere.
 *            - An EEPROM stream is used to store the LED state. On startup, this value is read;
 *              if the last stored state before power down was "on," the LED will automatically blink.
 */
#include "container/vector.h"
#include "driver/atmega328p/adc.h"
#include "driver/atmega328p/eeprom.h"
#include "driver/atmega328p/gpio.h"
#include "driver/atmega328p/serial.h"
#include "driver/atmega328p/timer.h"
#include "driver/atmega328p/watchdog.h"
#include "ml/lin_reg/lin_reg.h"
#include "target/system.h"

using namespace container;
using namespace driver::atmega328p;

namespace
{

// Vref = 5.0 V och 10-bitars ADC (0-1023)
static constexpr double VREF = 5.0;
static constexpr double ADC_MAX = 1023.0;

// ADC till volt
static inline double adcToVoltage(uint16_t raw) noexcept {
    return (static_cast<double>(raw) / ADC_MAX) * VREF;
}

// Volt till C för TMP36
static inline double voltageToTemp(double u_in) noexcept {
    return 100.0 * u_in - 50.0;
}

// Läs ADC, omvandla till volt, predicera temp med modellen och skriv ut
static inline void predictAndPrint(driver::atmega328p::Adc& adc,
                                   ml::lin_reg::LinReg& model,
                                   driver::atmega328p::Serial& serial) {
    const uint16_t raw = adc.read(driver::atmega328p::Adc::Pin::A0); 0-1023
    const double   u   = adcToVoltage(raw);     // volt
    const double   t   = model.predict(u);      // modellens prediktion i C
    serial.printf("ADC:%u  U_in:%.3f V  Pred: %.2f C\n", raw, u, t);
}


/** Pointer to the system implementation. */
target::System* mySys{nullptr};

/**
 * @brief Callback for the button.
 */
void buttonCallback() noexcept { mySys->handleButtonInterrupt(); }

/**
 * @brief Callback for the debounce timer.
 * 
 *        This callback is invoked whenever the debounce timer elapses.
 */
void debounceTimerCallback() noexcept { mySys->handleDebounceTimerInterrupt(); }

/**
 * @brief Callback for the toggle timer.
 * 
 *        This callback is invoked whenever the toggle timer elapses.
 */
void toggleTimerCallback() noexcept { mySys->handleToggleTimerInterrupt(); }

constexpr int round(const double number)
{
    // Round to the nearest integer - 2.7 is rounded to 3, 2.2 is rounded to 2, 
    // and -2.4 is rounded to 2.

    // number = 2.7 => we add 0.5 => 3.2. Then 3.2 is converted to int, i.e. the decimal part
    // truncated => the result is 3.
    return 0.0 <= number ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}

} // namespace

/**
 * @brief Initialize and run the system on the target MCU.
 * 
 * @return 0 on termination of the program (should never occur).
 */
int main()
{
    // Obtain a reference to the singleton serial device instance.
    auto& serial{Serial::getInstance()};
    serial.setEnabled(true);

    serial.printf("Machine learning project!\n");

    // Input voltage 0 - 5 V.
    const Vector<double> trainInput{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    // To till 7 - 12 till sådana.

    // Expected temperature in Celsius; T = 100 * Vin - 50.
    const Vector<double> trainOutput{-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0};

    ml::lin_reg::LinReg model{trainInput, trainOutput};

    // Träna modellen här.
    const bool trained = model.train(500, 0.1);
    

    for (const auto& input : trainInput)
    {
        const double prediction{model.predict(input)};
        const auto mV{input * 1000.0};

        serial.printf("Input: %d mV, predicted output: %d!\n", round(mV), round(prediction));
    }


    // Initialize the GPIO devices.
    Gpio led{8U, Gpio::Direction::Output};
    Gpio button{13U, Gpio::Direction::InputPullup, buttonCallback};

    // Initialize the timers.
    Timer debounceTimer{300U, debounceTimerCallback};
    Timer toggleTimer{100U, toggleTimerCallback};

    // Obtain a reference to the singleton watchdog timer instance.
    auto& watchdog{Watchdog::getInstance()};

    // Obtain a reference to the singleton EEPROM instance.
    auto& eeprom{Eeprom::getInstance()};

    // Obtain a reference to the singleton ADC instance.
    auto& adc{Adc::getInstance()};

    // Initialize the system with the given hardware.
    target::System system{led, button, debounceTimer, toggleTimer, serial, watchdog, eeprom, adc};
    mySys = &system;

    // Run the system perpetually on the target MCU.
    mySys->run();

    // This point should never be reached; the system is intended to run indefinitely on the target MCU.
    return 0;
}
