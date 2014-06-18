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

type BNode struct {
	activ float64 		// activation value
	thr float64			// threshold
	weights []float64	
	error float64	
}

// NewBNode creates new backpropagation network node.
func NewBNode(wCount int) *BNode {
	return &BNode{
		weights: make([]float64, wCount, wCount),	
	}
}
 
// Backprop main backpropagation network struct.
type Backprop struct {
	lhRate float64	// learning rate of the hidden layer
	loRate float64	// learning rate of the output layer

	input []*BNode	
	hidden []*BNode
	output []*BNode
	
	netInput []float64
	desiredOut []float64
}

// NewBackprop creates new backpropagation network with input, hidden and output layers.
func NewBackprop(inCount, hideCount, outCount int) *Backprop {
	n := &Backprop{
		lhRate: 0.15,
		loRate: 0.2,
		input: make([]*BNode, inCount, inCount),
		hidden: make([]*BNode, hideCount, hideCount),
		output: make([]*BNode, outCount, outCount),
		
	}
	rand.Seed(time.Now().Unix())
	for i:=0; i < inCount; i++ {
		n.input[i] = NewBNode(hideCount)
		for j:=0; j < hideCount; j++ {
			n.input[i].weights[j] = rand.Float64() - 0.49999
		}
	}
	
	for i:=0; i < hideCount; i++ {
		n.hidden[i] = NewBNode(outCount)
		for j:=0; j < outCount; j++ {
			n.hidden[i].weights[j] = rand.Float64()
		}		
	}
	for i:=0; i < outCount; i++ {
		n.output[i] = NewBNode(0)
	}
	
	// reset thresholds
	for i:=0; i < len(n.hidden); i++ {
		n.hidden[i].thr = rand.Float64()
	}
	for i:=0; i < len(n.output); i++ {
		n.output[i].thr = rand.Float64()
	}

	return n
}

// TrainingData holds single block of inputs and outputs for the training to run.
type TrainingData struct {
	Input []float64
	Output []float64
}

// Train performs network training for number of iterations, usually over 2000 iterations. 
func (n *Backprop) Train(iterations int, data []*TrainingData) {
	inputLen := len(n.input)
	outputLen := len(n.output)
	
	for i:=0; i < iterations; i++ {		
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

func (n *Backprop) calcActivation() {
	// a loop to set the activations of the hidden layer
	for h:=0; h < len(n.hidden); h++ {
		for i:=0; i < len(n.input); i++ {
			n.hidden[h].activ += n.netInput[i] * n.input[i].weights[h]
		}	
	}
	
	// calculate the output of the hidden
	for h:=0; h < len(n.hidden); h++ {
		n.hidden[h].activ += n.hidden[h].thr
		n.hidden[h].activ = sigmoid(n.hidden[h].activ)
	}
	
	// a loop to set the activations of the output layer
	for o:=0; o < len(n.output); o++ {
		for h:=0; h < len(n.hidden); h++ {
			n.output[o].activ += n.hidden[h].activ * n.hidden[h].weights[o]
		}		
	}
	
	// calculate the output of the output layer
	for o:=0; o < len(n.output); o++ {
		n.output[o].activ += n.output[o].thr
		n.output[o].activ = sigmoid(n.output[o].activ)
	}

}

// calcErrorOutput calculates error of each output neuron.
func (n *Backprop) calcErrorOutput() {
	for o:=0; o < len(n.output); o++ {
		n.output[o].error = n.output[o].activ * (1 - n.output[o].activ) * 
			(n.desiredOut[o] - n.output[o].activ)
	}
}

// calcErrorHidden calculate error of each hidden neuron.
func (n *Backprop) calcErrorHidden() {
	for h:=0; h < len(n.hidden); h++ {
		for o:=0; o < len(n.output); o++ {
			n.hidden[h].error += n.hidden[h].weights[o] * n.output[o].error
		}	
		n.hidden[h].error *= n.hidden[h].activ * (1 - n.hidden[h].activ)
	}
}

// calcNewThresholds calculate new thresholds for each neuron.
func (n *Backprop) calcNewThresholds() {
	// computing the thresholds for next iteration for hidden layer
	for h:=0; h < len(n.hidden); h++ {
		n.hidden[h].thr += n.hidden[h].error * n.lhRate	
	}
	// computing the thresholds for next iteration for output layer
	for o:=0; o < len(n.output); o++ {
		n.output[o].thr += n.output[o].error * n.loRate
	}

}

// calcNewWeightsHidden calculate new weights between hidden and output.
func (n *Backprop) calcNewWeightsHidden() {
	for h:=0; h < len(n.hidden); h++ {
		temp := n.hidden[h].activ * n.loRate
		for o:=0; o < len(n.output); o++ {
			n.hidden[h].weights[o] += temp * n.output[o].error
		}
	}
}

// calcNewWeightsInput .
func (n *Backprop) calcNewWeightsInput() {
	for i:=0; i < len(n.netInput); i++ {
		temp := n.netInput[i] * n.lhRate
		for h:=0; h < len(n.hidden); h++ {
			n.input[i].weights[h] += temp * n.hidden[h].error
		}
	}
}

// calcTotalErrorPattern.
func (n *Backprop) calcTotalError() float64 {
	temp := 0.0
	for o:=0; o < len(n.output); o++ {
		temp += n.output[o].error
	}
	return temp
}

// Predict calculates network output based on provided input, returns raw float64 activation value.
func (n *Backprop) Predict(input []float64) []float64 {
	n.netInput = input
	n.calcActivation()
	out := make([]float64, len(n.output), len(n.output))
	for i, node := range n.output {
		out[i] = node.activ
	}
	return out
}

// PredictInt calculates network output based on provided input, this is main method to call after Train.
func (n *Backprop) PredictInt(input []float64) []int {
	n.netInput = input
	n.calcActivation()
	out := make([]int, len(n.output), len(n.output))
	for i, node := range n.output {
		if node.activ > 0.5 {
			out[i] = 1
		}
	}
	return out
}


