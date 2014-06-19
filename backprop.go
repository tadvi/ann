// Artificial Neural Networks (ann) library in Go
// Backpropagation Network - Backprop
// Various sources has been used to create this Neural Network
// Credits are to Yosif Mohammed and Mike Gold
// Implemetation in Go by Tad Vizbaras
// released under MIT license
package ann

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// sigmoid helper function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// BNode node for backpropagation training based network
type BNode struct {
	Thr     float64 // threshold
	Weights []float64

	activ   float64 // activation value
	error   float64
}

// NewBNode creates new backpropagation network node.
func NewBNode(wCount int) *BNode {
	return &BNode{
		Weights: make([]float64, wCount, wCount),
	}
}

// Backprop main backpropagation network.
// Public members can be persisted to json or database.
type Backprop struct {
	Input  []*BNode
	Hidden []*BNode
	Output []*BNode

	lhRate float64 // learning rate of the hidden layer
	loRate float64 // learning rate of the output layer
	
	netInput   []float64
	desiredOut []float64
}

// NewBackprop creates new backpropagation network with input, hidden and output layers.
func NewBackprop(inCount, hideCount, outCount int) *Backprop {
	n := &Backprop{
		lhRate: 0.15,
		loRate: 0.2,
		Input:  make([]*BNode, inCount, inCount),
		Hidden: make([]*BNode, hideCount, hideCount),
		Output: make([]*BNode, outCount, outCount),
	}
	rand.Seed(time.Now().Unix())
	for i := 0; i < inCount; i++ {
		n.Input[i] = NewBNode(hideCount)
		for j := 0; j < hideCount; j++ {
			n.Input[i].Weights[j] = rand.Float64() - 0.49999
		}
	}

	for i := 0; i < hideCount; i++ {
		n.Hidden[i] = NewBNode(outCount)
		for j := 0; j < outCount; j++ {
			n.Hidden[i].Weights[j] = rand.Float64()
		}
	}
	for i := 0; i < outCount; i++ {
		n.Output[i] = NewBNode(0)
	}

	// reset thresholds
	for i := 0; i < len(n.Hidden); i++ {
		n.Hidden[i].Thr = rand.Float64()
	}
	for i := 0; i < len(n.Output); i++ {
		n.Output[i].Thr = rand.Float64()
	}

	return n
}

// TrainingData holds single block of inputs and outputs for the training to run.
type TrainingData struct {
	Input  []float64
	Output []float64
}

// Train performs network training for number of iterations, usually over 2000 iterations.
func (n *Backprop) Train(iterations int, data []*TrainingData) {
	inputLen := len(n.Input)
	outputLen := len(n.Output)

	for i := 0; i < iterations; i++ {
		for _, tr := range data {
			if inputLen != len(tr.Input) {
				panic(fmt.Sprintf("expected training data input length %d got %d", inputLen, len(tr.Input)))
			}
			if outputLen != len(tr.Output) {
				panic(fmt.Sprintf("expected traing data output length %d got %d", outputLen, len(tr.Output)))
			}
			n.netInput = tr.Input
			n.desiredOut = tr.Output
			n.TrainOnePattern()
		}
	}

}

// TrainOnePattern train single pattern.
func (n *Backprop) TrainOnePattern() {
	n.calcActivation()
	n.calcErrorOutput()
	n.calcErrorHidden()
	n.calcNewThresholds()
	n.calcNewWeightsHidden()
	n.calcNewWeightsInput()
}

// SetLearningRate sets learning rate for the backpropagation.
func (n *Backprop) SetLearningRates(lhRate, loRate float64) {
	n.lhRate = lhRate
	n.loRate = loRate
}

func (n *Backprop) calcActivation() {
	// a loop to set the activations of the hidden layer
	for h := 0; h < len(n.Hidden); h++ {
		for i := 0; i < len(n.Input); i++ {
			n.Hidden[h].activ += n.netInput[i] * n.Input[i].Weights[h]
		}
	}

	// calculate the output of the hidden
	for h := 0; h < len(n.Hidden); h++ {
		n.Hidden[h].activ += n.Hidden[h].Thr
		n.Hidden[h].activ = sigmoid(n.Hidden[h].activ)
	}

	// a loop to set the activations of the output layer
	for o := 0; o < len(n.Output); o++ {
		for h := 0; h < len(n.Hidden); h++ {
			n.Output[o].activ += n.Hidden[h].activ * n.Hidden[h].Weights[o]
		}
	}

	// calculate the output of the output layer
	for o := 0; o < len(n.Output); o++ {
		n.Output[o].activ += n.Output[o].Thr
		n.Output[o].activ = sigmoid(n.Output[o].activ)
	}

}

// calcErrorOutput calculates error of each output neuron.
func (n *Backprop) calcErrorOutput() {
	for o := 0; o < len(n.Output); o++ {
		n.Output[o].error = n.Output[o].activ * (1 - n.Output[o].activ) *
			(n.desiredOut[o] - n.Output[o].activ)
	}
}

// calcErrorHidden calculate error of each hidden neuron.
func (n *Backprop) calcErrorHidden() {
	for h := 0; h < len(n.Hidden); h++ {
		for o := 0; o < len(n.Output); o++ {
			n.Hidden[h].error += n.Hidden[h].Weights[o] * n.Output[o].error
		}
		n.Hidden[h].error *= n.Hidden[h].activ * (1 - n.Hidden[h].activ)
	}
}

// calcNewThresholds calculate new thresholds for each neuron.
func (n *Backprop) calcNewThresholds() {
	// computing the thresholds for next iteration for hidden layer
	for h := 0; h < len(n.Hidden); h++ {
		n.Hidden[h].Thr += n.Hidden[h].error * n.lhRate
	}
	// computing the thresholds for next iteration for output layer
	for o := 0; o < len(n.Output); o++ {
		n.Output[o].Thr += n.Output[o].error * n.loRate
	}

}

// calcNewWeightsHidden calculate new weights between hidden and output.
func (n *Backprop) calcNewWeightsHidden() {
	for h := 0; h < len(n.Hidden); h++ {
		temp := n.Hidden[h].activ * n.loRate
		for o := 0; o < len(n.Output); o++ {
			n.Hidden[h].Weights[o] += temp * n.Output[o].error
		}
	}
}

// calcNewWeightsInput .
func (n *Backprop) calcNewWeightsInput() {
	for i := 0; i < len(n.netInput); i++ {
		temp := n.netInput[i] * n.lhRate
		for h := 0; h < len(n.Hidden); h++ {
			n.Input[i].Weights[h] += temp * n.Hidden[h].error
		}
	}
}

// calcTotalErrorPattern.
func (n *Backprop) calcTotalError() float64 {
	temp := 0.0
	for o := 0; o < len(n.Output); o++ {
		temp += n.Output[o].error
	}
	return temp
}

// Predict calculates network output based on provided input, returns raw float64 activation value.
func (n *Backprop) Predict(input []float64) []float64 {
	n.netInput = input
	n.calcActivation()
	out := make([]float64, len(n.Output), len(n.Output))
	for i, node := range n.Output {
		out[i] = node.activ
	}
	return out
}

// PredictInt calculates network output based on provided input, this is main method to call after Train.
func (n *Backprop) PredictInt(input []float64) []int {
	n.netInput = input
	n.calcActivation()
	out := make([]int, len(n.Output), len(n.Output))
	for i, node := range n.Output {
		if node.activ > 0.5 {
			out[i] = 1
		}
	}
	return out
}
