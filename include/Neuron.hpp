#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
using namespace std;

class Neuron
{
    public:

        Neuron(double val);

        // Fast Sigmoid Function
        // f(x) = x / (1 + |x|)
        void activate();

        // Derivative for fast sigmoid function
        // f'(x) = f(x) * (1 - f(x))

        void derive();

        // Getter
        double getVal() { return this-> val; }
        double getActivatedVal() { return this->derivedVal; }
        double getDerivedVal() { return this->derievedVal; }

    private:

        // 1.5
        double val;

        // 0-1
        double activateVal;

        double derivedVal;
};

#endif

