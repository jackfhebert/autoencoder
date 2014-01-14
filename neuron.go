/*
A single neuron, can backprop itself to learn weights.
*/

import (
       "rand"
)

package AutoEncoder

type Neuron struct {
    alpha float64
    weights []float64
    inputs chan<- float64
    outputs <-chan float64
}

func NewNeuron(numInputs int) {
  node := &Neuron(0.01, make([]float64, numInputs), nil, nil)
  for i:= 0; i < numInputs; i++ {
    weights[i] = rand.Float64()
  }
}

func (node *Neuron) Process() {
  // This should weight for either channel to have a value, but only
  // if the channels exist.
  
}

// Compute the weighted logistic function of the node.
func computeOutput(input, weight float64) float64 {
  return weight * (1.0 / (1 + math.Exp(-weight)))
}

// Compute the activation of this node.
func (node* Neuron) Predict(input []float64) float64 {
  value := 0.0
  for i := 0; i < len(node.weights); i++ {
    value += computeOutput(input[i], node.weights[i])
  }

  return 0.0
}

func (node* Neuron) Update(input []float64, result float64) {
  // Figure out what this node would output.
  wouldOutput = node.Predict(input)
  // Update the weight per input to get closer to the desired output.
  for i := 0; i < len(node.weights); i++ {
    node.weights[i] += node.alpha * (result - wouldOutput) * input[i]
  }
}