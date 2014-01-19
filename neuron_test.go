package AutoEncoder

import (
	"fmt"
	"math/rand"
	"testing"
)

// Try to learn the logical OR function over three
// inputs.
func Test_LearnOr(t *testing.T) {
	neuron := NewNeuron(3)
	inputs := []float64{0, 0, 0}
	result := 0.0
	for i := 0; i < 50000; i++ {
		// Each iteration, randomize the inputs.
		result = 0
		for j := 0; j < len(inputs); j++ {
			if rand.Float64() < .5 {
				inputs[j] = 0
			} else {
				inputs[j] = 1
				result = 1
			}
		}

		neuron.Update(inputs, result)
	}
	neuron.PrintDebugString("OR")

	inputs = []float64{1, 1, 1}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on all 1's")
	}

	inputs = []float64{1, 0, 1}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 2 1's")
	}

	inputs = []float64{0, 1, 1}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 2 1's")
	}

	inputs = []float64{1, 1, 0}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 2 1's")
	}

	inputs = []float64{1, 0, 0}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 1 1's")
	}

	inputs = []float64{0, 1, 0}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 1 1's")
	}

	inputs = []float64{0, 0, 1}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on 1 1's")
	}

	inputs = []float64{0, 0, 0}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 0 1's")
		fmt.Println(neuron.weights)
		fmt.Println(neuron.Predict(inputs))
	}
}

// Try to learn the logical AND over three inputs.
func Test_LearnAnd(t *testing.T) {
	neuron := NewNeuron(3)
	inputs := []float64{0, 0, 0}
	result := 0.0
	for i := 0; i < 150000; i++ {
		// Each iteration, randomize the inputs.
		result = 1
		for j := 0; j < len(inputs); j++ {
			if rand.Float64() < .5 {
				inputs[j] = 0
				result = 0
			} else {
				inputs[j] = 1
			}
		}

		neuron.Update(inputs, result)
	}
	neuron.PrintDebugString("AND")

	inputs = []float64{1, 1, 1}
	if neuron.Predict(inputs) < .5 {
		t.Error("failed on all 1's")
		fmt.Println(neuron.weights)
	}

	inputs = []float64{1, 0, 1}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 2 1's")
		fmt.Println(neuron.weights)
	}

	inputs = []float64{0, 1, 1}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 2 1's")
		fmt.Println(neuron.weights)
	}

	inputs = []float64{1, 1, 0}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 2 1's")
		fmt.Println(neuron.weights)
	}

	inputs = []float64{1, 0, 0}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 1 1's")
		fmt.Println(neuron.weights)
	}

	inputs = []float64{0, 1, 0}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 1 1's")
	}

	inputs = []float64{0, 0, 1}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 1 1's")
	}

	inputs = []float64{0, 0, 0}
	if neuron.Predict(inputs) > .5 {
		t.Error("failed on 0 1's")
	}
}

//
func Test_LearnLayerAndOrAnd(t *testing.T) {
	layer := NewNeuronLayer(2, 3)
	inputs := []float64{0, 0}
	outputs := []float64{0, 0, 0}

	for i := 0; i < 150000; i++ {
		// Each iteration, randomize the inputs.
		inputs = []float64{0, 0}
		outputs = []float64{0, 0, 0}
		layer.Update(inputs, outputs)
		inputs = []float64{0, 1}
		outputs = []float64{0, 1, 0}
		layer.Update(inputs, outputs)
		inputs = []float64{1, 0}
		outputs = []float64{0, 1, 0}
		layer.Update(inputs, outputs)
		inputs = []float64{1, 1}
		outputs = []float64{1, 1, 1}
		layer.Update(inputs, outputs)
	}
	layer.PrintDebugString("AndOrAnd")

	inputs = []float64{0, 0}
	outputs = layer.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on all 1's")
	}
	if outputs[1] > .5 {
		t.Error("failed on all 1's")
	}
	if outputs[2] > .5 {
		t.Error("failed on all 1's")
	}

	inputs = []float64{0, 1}
	outputs = layer.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on all 1's")
	}
	if outputs[1] < .5 {
		t.Error("failed on all 1's")
	}
	if outputs[2] > .5 {
		t.Error("failed on all 1's")
	}

	inputs = []float64{1, 0}
	outputs = layer.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on all 1's")
	}
	if outputs[1] < .5 {
		t.Error("failed on all 1's")
	}
	if outputs[2] > .5 {
		t.Error("failed on all 1's")
	}

	inputs = []float64{1, 1}
	outputs = layer.Predict(inputs)
	if outputs[0] < .5 {
		t.Error("failed on all 1's")
	}
	if outputs[1] < .5 {
		t.Error("failed on all 1's")
	}
	if outputs[2] < .5 {
		t.Error("failed on all 1's")
	}
}

func Test_StackLearnAndOrAnd(t *testing.T) {
	dimensions := []int{2, 2, 3}
	stack := NewStackedNet(dimensions)
	// Expect this to print 3 neurons in the first layer, 3 in the second layer.
	// The first ones should have 3 weights (2 inputs, 1 bias) and the second
	// should have 4 weights (3 nodes, 1 bias).
	inputs := []float64{0, 0}
	outputs := []float64{0, 0, 0}

	for i := 0; i < 3500; i++ {
		// Each iteration, randomize the inputs.
		inputs = []float64{0, 0}
		outputs = []float64{0, 0, 0}
		stack.Update(inputs, outputs)
		inputs = []float64{0, 1}
		outputs = []float64{0, 1, 0}
		stack.Update(inputs, outputs)
		inputs = []float64{1, 0}
		outputs = []float64{0, 1, 0}
		stack.Update(inputs, outputs)
		inputs = []float64{1, 1}
		outputs = []float64{1, 1, 1}
		stack.Update(inputs, outputs)
	}
	stack.PrintDebugString("AndOrAndStack")

	inputs = []float64{0, 0}
	outputs = stack.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on all 0's", outputs[0])
	}
	if outputs[1] > .5 {
		t.Error("failed on all 0's", outputs[1])
	}
	if outputs[2] > .5 {
		t.Error("failed on all 0's")
	}

	inputs = []float64{0, 1}
	outputs = stack.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on some 1's")
	}
	if outputs[1] < .5 {
		t.Error("failed on some 1's")
	}
	if outputs[2] > .5 {
		t.Error("failed on some 1's")
	}

	inputs = []float64{1, 0}
	outputs = stack.Predict(inputs)
	if outputs[0] > .5 {
		t.Error("failed on some 1's")
	}
	if outputs[1] < .5 {
		t.Error("failed on some 1's")
	}
	if outputs[2] > .5 {
		t.Error("failed on some 1's")
	}

	inputs = []float64{1, 1}
	outputs = stack.Predict(inputs)
	if outputs[0] < .5 {
		t.Error("failed on all 1's", outputs[0])
	}
	if outputs[1] < .5 {
		t.Error("failed on all 1's", outputs[1])
	}
	if outputs[2] < .5 {
		t.Error("failed on all 1's", outputs[2])
	}

}

func Test_StackConstructor(t *testing.T) {
	dimensions := []int{3, 2, 1}
	stack := NewStackedNet(dimensions)
	// Expect this to print 2 neurons in the first layer, 1 in the second layer.
	// The first ones should have 4 weights (3 inputs, 1 bias) and the second
	// should have 3 weights (2 nodes, 1 bias).
	stack.PrintDebugString("test construction")
}
