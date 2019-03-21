#include "../include/Neuron.hpp"

// Constructor
Neuron::Neuron(double val) {
    this->val = val;
    activate();
    derive();
}

// Fast Sigmoid Function
// f(x) = x / (1 + |x|)
void Neuron::activate() {
    this->activatedVal = this->val / (1 + Math.abs(this->val);
}


// Derivative
void Neuron::derive() {
    this->derivedVal;
}
