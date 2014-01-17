/*
A single neuron, can backprop itself to learn weights.
*/

package AutoEncoder

import (
	"math"
	"math/rand"
)

type Neuron struct {
	alpha   float64
	decay   float64
	weights []float64
}

type NeuronLayer struct {
     nodes []*Neuron
}

func NewNeuonLayer(numInputs, numNeurons int) *NeuronLayer {
  layer := &NeuronLayer{make([]*Neuron, numNeurons)}
  for i := 0; i < len(layer.nodes); i++ {
    layer.nodes[i] = NewNeuron(numInputs)
  }
  return layer
}

func NewNeuron(numInputs int) *Neuron {
	// Want the number of weights to be 1 longer than the number
	// of inputs - this simulates having a bias unit- sorta like
	// that silly +C back when you did calculus.
	node := &Neuron{0.2, 0.01, make([]float64, numInputs+1)}
	for i := 0; i < len(node.weights); i++ {
		node.weights[i] = rand.Float64()*.5 - 0.25
	}
	return node
}


func (layer *NeuronLayer) Predict(input []float64) []float64 {
  result := make([]float64, len(layer.nodes))
  for i := 0; i < len(layer.nodes); i++ {
    result[i] = layer.nodes[i].Predict(input)
  }
  return result
}


// Compute the activation of this node.
func (node *Neuron) Predict(input []float64) float64 {
	// Sum up the weighted activation.
	value := 0.0
	// Be careful that node.weights is longer than len(input).
	for i := 0; i < len(input); i++ {
		value += input[i] * node.weights[i]
	}
	// Add in the activation for the bias unit.
	value += 1 * node.weights[len(node.weights)-1]

	return 1.0 / (1 + math.Exp(-value))
}

func updateWeight(input, error, weight float64, node *Neuron) float64 {
     // This is the gradient of the error function.
     grad := error * input
     // Add in a penalty for large weights.
     decay := node.decay * weight
     return -1 * node.alpha * (grad + decay)
}

func (layer *NeuronLayer) Update(input, result []float64, 

func (node *Neuron) Update(input []float64, result float64) float64 {
	// Figure out what this node would output.
	wouldOutput := node.Predict(input)
	error := wouldOutput - result
	node.updateByError(input, error)
	return error
}

func (node *Neuron) updateByError(input []float64, error float64) {
	// Update the weight per input to get closer to the desired output.
	for i := 0; i < len(input); i++ {
		node.weights[i] += updateWeight(
				input[i], error, node.weights[i], node)

	}
        index := len(node.weights) - 1
	node.weights[index] += updateWeight(
		  1, error, node.weights[index], node)
}