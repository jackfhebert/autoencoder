/*
A single neuron, can backprop itself to learn weights.
*/

package AutoEncoder

import (
       "math"
       "math/rand"
)

type Neuron struct {
    alpha float64
    decay float64
    weights []float64
    inputs chan<- float64
    outputs <-chan float64
}

func NewNeuron(numInputs int) *Neuron {
  // Want the number of weights to be 1 longer than the number
  // of inputs - this simulates having a bias unit- sorta like
  // that silly +C back when you did calculus.
  node := &Neuron{0.05, 0.005, make([]float64, numInputs + 1), nil, nil}
  for i:= 0; i < len(node.weights); i++ {
    node.weights[i] = rand.Float64() * .5 - 0.25
  }
  return node
}

func (node *Neuron) Process() {
  // This should weight for either channel to have a value, but only
  // if the channels exist.
  
}

// Compute the activation of this node.
func (node* Neuron) Predict(input []float64) float64 {
  // Sum up the weighted activation.
  value := 0.0
  // Be careful that node.weights is longer than len(input).
  for i := 0; i < len(input); i++ {
    value += input[i] * node.weights[i]
  }
  // Add in the activation for the bias unit.
  value += 1 * node.weights[len(node.weights) - 1]

  return 1.0 / (1 + math.Exp(-value))
}

func (node* Neuron) Update(input []float64, result float64) float64 {
  // Figure out what this node would output.
  wouldOutput := node.Predict(input)

  // Update the weight per input to get closer to the desired output.
  for i := 0; i < len(input); i++ {
    // This is the gradient of the error function.
    grad := (wouldOutput - result) * input[i]
    // Add in a penalty for large weights.
    decay := node.decay * node.weights[i]
    node.weights[i] -= node.alpha * (grad + decay)
  }

  // Update the bias node which always has an input value of 1.
  grad := (wouldOutput - result)
  decay := node.decay * node.weights[len(node.weights) - 1]

  node.weights[len(node.weights) - 1] -= node.alpha * (grad + decay)
  return result - wouldOutput
}