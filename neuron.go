/*
Neuron: A single neuron, can backprop itself to learn weights.
NeuronLayer: N inputs to K neurons. So takes N dimensional input to K
  dimensional output. There are no size requirments between N and K.
StackedNet: takes N dimensional input to K dimensional output with M
  layers of various dimensions.
*/

package AutoEncoder

import (
	//"fmt"
	"math"
	"math/rand"
	"strconv"
)

type Neuron struct {
	// The learning rate for the neuron. Controls how quickly it
	// proceeds through gradient descence.
	alpha float64
	// Penalty value for large weights to perform some amount of
	// regularization.
	decay float64
	// The weights for the N inputs to this node. Should be one more
	// weight than inputs to handle the bias unit. That weight will be
	// last in the list.
	weights []float64
}

// Wrapper structure and logic for a set of neurons fully connected to a
// set of input. This allows learning K functions over N inputs.
type NeuronLayer struct {
	// The list of neurons comprising this layer.
	nodes []*Neuron
}

// Wrapper structure around an ordered list of NeuronLayers.
// The layers are connected together in order with the 0th layer connected
// to the inputs, the 1st to the 0th and so on.
type StackedNet struct {
	layers []*NeuronLayer
}

// Create a new StackedNet of the given dimensions. Passing a list like
// {10, 20, 20} would say that the input is of 10 dimensions and will create
// two NeuronLayers each with 20 dimensions.
// Another valid choid would be {10, 20, 2} where the hidden layer has 20
// dimensions but the final output only has 2.
func NewStackedNet(dimensions []int) *StackedNet {
	// The first dimensions specifies the initial input, it
	// isn't a layer in the stack.
	stack := &StackedNet{make([]*NeuronLayer, len(dimensions)-1)}
	// The input dimension of each subsequent layer is equal to the
	// number of nodes in the previous layer.
	previousSize := dimensions[0]
	for i := 1; i < len(dimensions); i++ {
		currentSize := dimensions[i]
		stack.layers[i-1] = NewNeuronLayer(previousSize, currentSize)
		previousSize = currentSize
	}
	return stack
}

// Create a new NeuronLayer with the given number of neurons, each ready to
// deal with N inputs.
func NewNeuronLayer(numInputs, numNeurons int) *NeuronLayer {
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
	node := &Neuron{0.05, 0.0, make([]float64, numInputs+1)}
	for i := 0; i < len(node.weights); i++ {
		// Initialize the weights on the domain [-.25, .25].
		// I have no yet how much the scale matters on init.
		node.weights[i] = rand.Float64()*.5 - 0.25
	}
	return node
}

func (stack *StackedNet) Predict(input []float64) []float64 {
	// Walk through the layers, passing the intermediate activations
	// through as inputs to the next layer.
	workingInputs := input
	for i := 0; i < len(stack.layers); i++ {
		workingInputs = stack.layers[i].Predict(workingInputs)
	}
	return workingInputs
}

func (layer *NeuronLayer) Predict(input []float64) []float64 {
	// Build the array to hold the final results, one per node.
	result := make([]float64, len(layer.nodes))
	// Compute each nodes output.
	for i := 0; i < len(layer.nodes); i++ {
		result[i] = layer.nodes[i].Predict(input)
		layer.nodes[i].PrintDebugString("   ")
		//fmt.Println("  input", input, "result", result[i])
	}
	return result
}

// Compute the activation of this node.
func (node *Neuron) Predict(input []float64) float64 {
	// Sum up the weighted activation of each input signal against
	// the weight for that input.
	value := 0.0
	// Be careful that node.weights is longer than len(input).
	for i := 0; i < len(input); i++ {
		value += input[i] * node.weights[i]
	}
	// Add in the activation for the bias unit.
	value += 1 * node.weights[len(node.weights)-1]

	// Sigmoid ranges from (0, 1) as the value tends to +-\infty.
	return 1.0 / (1 + math.Exp(-value))
}

// Compute the update delta for the weight of this neuron given the
// input, error and current weight.
func updateWeight(input, error, weight float64, node *Neuron) float64 {
	// This is the gradient of the error function.
	grad := error * input
	// Add in a penalty for large weights. aka regularization.
	decay := node.decay * weight
	return -1 * node.alpha * (grad + decay)
}

// Update the weights for all of the nodes in a stack net.
// This is error backpropogation.
func (stack *StackedNet) Update(input, target []float64) {
	inputsByLayer := make([][]float64, len(stack.layers)+1)
	inputsByLayer[0] = input
	//fmt.Println("input", inputsByLayer[0], "target", target)
	for i := 0; i < len(stack.layers); i++ {
		inputsByLayer[i+1] = stack.layers[i].Predict(inputsByLayer[i])
		//fmt.Println("next layer:", inputsByLayer[i + 1])

	}
	lastLayer := len(stack.layers) - 1
	//fmt.Println("target was:", target, "inputs by layer", inputsByLayer)

	error := stack.layers[lastLayer].Update(inputsByLayer[lastLayer], target)
	//fmt.Println("Last layer after update:", stack.layers[lastLayer].Predict(inputsByLayer[lastLayer]))

	for i := 0; i < len(error); i++ {
		activation := inputsByLayer[lastLayer][i]
		error[i] = error[i] * activation * (1.0 - activation)
	}
	//fmt.Println("target", target, "activation", inputsByLayer[lastLayer])
	//fmt.Println("initial error", error)

	for i := len(stack.layers) - 2; i >= 0; i-- {
		error = stack.layers[i].updateByError(inputsByLayer[i], error)
		//fmt.Println("layer after update:", stack.layers[i].Predict(inputsByLayer[i]))

		//fmt.Println("layer after update:", stack.layers[i+1].Predict(stack.layers[i].Predict(inputsByLayer[i])))

		//fmt.Println("error before", error)
		//fmt.Println("layer activation", inputsByLayer[i])
		for j := 0; j < len(error); j++ {
			activation := inputsByLayer[i][j]
			error[j] = error[j] * activation * (1.0 - activation)
		}
		//fmt.Println("error", error)
	}
}

// Update all of the nodes in a layer for a given input and output vector.
func (layer *NeuronLayer) Update(input, target []float64) []float64 {
	// Get all of the outputs for the layer.
	layerOutput := layer.Predict(input)
	// Build up an array of the errors per node measures from the provided
	// target output.
	nodeError := make([]float64, len(layer.nodes))
	for i := 0; i < len(layer.nodes); i++ {
		nodeError[i] = layerOutput[i] - target[i]
	}
	// Pass that off to a function to update the weights.
	return layer.updateByError(input, nodeError)
}

// Update all of the nodes in a layer for a given input and error vector.
// Returns the weighted error per input. Note that this is merged from
// nodes in the layer by summation.
func (layer *NeuronLayer) updateByError(input, error []float64) []float64 {
	// Want to return a merged error to the inputs. This makes most sense
	// in a stacked net and very little in a single layer.
	mergedInputError := make([]float64, len(input))
	for i := 0; i < len(mergedInputError); i++ {
		mergedInputError[i] = 0
	}

	// For each node, update it for the provided error value. This takes all
	//  of the inputs that the node was given, so it updates the set of weights.
	for i := 0; i < len(layer.nodes); i++ {
		nodeError := error[i]
		weightedInputError := layer.nodes[i].updateByError(input, nodeError)
		// Merge the input error terms from this node into the array.
		for j := 0; j < len(weightedInputError); j++ {
			mergedInputError[j] += weightedInputError[j]
		}
	}
	return mergedInputError
}

// Update the weights for a neuron given the input vector and target
// output value.
func (node *Neuron) Update(input []float64, result float64) float64 {
	// Figure out what this node would output.
	wouldOutput := node.Predict(input)
	error := wouldOutput - result
	node.updateByError(input, error)
	return error
}

// Update the weights for a neuron given the input vector and pre-computed
// error. Returns the weighted error per input.
func (node *Neuron) updateByError(input []float64, error float64) []float64 {
	// Update the weight per input to get closer to the desired output.
	weightedError := make([]float64, len(input))
	for i := 0; i < len(input); i++ {
		//fmt.Println("at node error", error, "weight", node.weights[i])
		weightedError[i] = error * node.weights[i]
		node.weights[i] += updateWeight(
			input[i], error, node.weights[i], node)

	}
	index := len(node.weights) - 1
	node.weights[index] += updateWeight(
		1, error, node.weights[index], node)
	return weightedError
}

// Print some debug info about this neuron.
func (node *Neuron) PrintDebugString(prefix string) {
	//fmt.Println(prefix, node.weights)
}

// Print some debug info about this layer, notable the neurons it contains.
func (layer *NeuronLayer) PrintDebugString(prefix string) {
	for i := 0; i < len(layer.nodes); i++ {
		layer.nodes[i].PrintDebugString(prefix)
	}
}

// Print some debug info about a stack, with notes for each layer.
func (stack *StackedNet) PrintDebugString(prefix string) {
	for i := 0; i < len(stack.layers); i++ {
		stack.layers[i].PrintDebugString(prefix + ":layer_" + strconv.Itoa(i))
	}
}
