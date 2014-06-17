// Artificial Neural Networks (ann) library in Go
// Self-Organizing Maps - SOM
// Credits to Paras Chopra for initial implementation in Python
// Implemetation in Go by Tad Vizbaras
// released under MIT license
package ann

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Node is single node in SOM network.
type Node struct {
	fvSize int
	pvSize int
	fv     []float64
	pv     []float64
	x      int
	y      int
}

// NewNode create new node.
func NewNode(fvSize, pvSize, y, x int) *Node {
	node := &Node{
		fvSize: fvSize,
		pvSize: pvSize,
		y:      y,
		x:      x,
		fv:     make([]float64, fvSize, fvSize),
		pv:     make([]float64, pvSize, pvSize),
	}

	rand.Seed(time.Now().Unix())
	for i := 0; i < fvSize; i++ {
		node.fv[i] = rand.Float64()
	}
	for i := 0; i < pvSize; i++ {
		node.pv[i] = rand.Float64()
	}
	return node
}

// String pretty print node information.
func (node Node) String() string {
	var buff bytes.Buffer

	buff.WriteString("Node FV [")
	for i := 0; i < node.fvSize; i++ {
		buff.WriteString(fmt.Sprintf("%.2f, ", node.fv[i]))
	}
	buff.WriteString("] PV [")
	for i := 0; i < node.pvSize; i++ {
		buff.WriteString(fmt.Sprintf("%.2f, ", node.pv[i]))
	}
	buff.WriteString("]\n")
	return buff.String()
}

// SOM self-orginizing map.
type SOM struct {
	height       int
	width        int
	radius       int
	total        int
	learningRate float64
	nodes        []*Node
	fvSize       int
	pvSize       int
}

// NewSOM creates new self organizing map with specific width and height.
func NewSOM(height, width, fvSize, pvSize int) *SOM {
	total := height * width
	som := &SOM{
		height:       height,
		width:        width,
		radius:       (height + width) / 2,
		total:        total,
		learningRate: 0.05,
		fvSize:       fvSize,
		pvSize:       pvSize,
		nodes:        make([]*Node, total, total),
	}

	// fill SOM network with nodes
	for i := 0; i < som.height; i++ {
		for j := 0; j < som.width; j++ {
			som.nodes[i*som.width+j] = NewNode(som.fvSize, som.pvSize, i, j)
		}
	}
	return som
}

// Train performs SOM training for specified number of iterations.
func (som *SOM) Train(iterations int, fvInputTrain [][]float64, pvInputTrain [][]float64) {
	// helper type for storing calculated values
	type StackValue struct {
		k      int
		fvTemp []float64
		pvTemp []float64
	}

	if len(fvInputTrain) != len(pvInputTrain) {
		panic("length of fvInputTrain should match pvInputTrain")
	}

	timeConstant := float64(iterations) / math.Log(float64(som.radius))
	radiusDecaying := 0.0
	lrd := 0.0 // learning rate decaying
	influence := 0.0
	stack := []*StackValue{}
	length := len(fvInputTrain)

	for i := 1; i < iterations+1; i++ {
		radiusDecaying = float64(som.radius) * math.Exp(float64(-1.0*i)/timeConstant)
		lrd = som.learningRate * math.Exp(float64(-1.0*i)/timeConstant)

		for j := 0; j < length; j++ {
			fvInput := fvInputTrain[j]
			pvInput := pvInputTrain[j]
			best := som.bestMatch(fvInput)
			stack = []*StackValue{}

			for k := 0; k < som.total; k++ {
				dist := som.distance(som.nodes[best], som.nodes[k])
				if dist < radiusDecaying {
					fvTemp := []float64{}
					pvTemp := []float64{}
					influence = math.Exp((-1.0 * math.Pow(dist, 2)) / (2 * radiusDecaying * float64(i)))

					// perform FV learning
					for m := 0; m < som.fvSize; m++ {
						adjustment := influence * lrd * (fvInput[m] - som.nodes[k].fv[m])
						fvTemp = append(fvTemp, som.nodes[k].fv[m]+adjustment)
					}

					// perform PV learning
					for m := 0; m < som.pvSize; m++ {
						adjustment := influence * lrd * (pvInput[m] - som.nodes[k].pv[m])
						pvTemp = append(pvTemp, som.nodes[k].pv[m]+adjustment)
					}

					// push node onto stack to update in the next interval
					stack = append(stack, &StackValue{k, fvTemp, pvTemp})

				}
			}

			// update nodes with new learned values
			stackLen := len(stack)
			for k := 0; k < stackLen; k++ {
				som.nodes[stack[k].k].fv = stack[k].fvTemp
				som.nodes[stack[k].k].pv = stack[k].pvTemp
			}
		}
	}
}

// predict performs prediction for SOM.
func (som *SOM) predict(fv []float64) []float64 {
	best := som.bestMatch(fv)
	return som.nodes[best].pv
}

// predictInt performs prediction for SOM and rounds resulting values to percentage.
func (som *SOM) predictInt(fv []float64) []int {
	best := som.bestMatch(fv)
	res := []int{}
	for _, val := range som.nodes[best].pv {
		res = append(res, int(val*100))
	}
	return res
}

// bestMatch find best matching node index.
func (som SOM) bestMatch(fvTarget []float64) int {
	minimum := math.Sqrt(float64(som.fvSize))
	minimumIndex := 1
	for i := 0; i < som.total; i++ {
		temp := som.fvDistance(som.nodes[i].fv, fvTarget)
		if temp < minimum {
			minimum = temp
			minimumIndex = i
		}
	}
	return minimumIndex
}

// fvDistance calculates distance of two vectors.
func (som SOM) fvDistance(fv1, fv2 []float64) float64 {
	temp := 0.0
	for j := 0; j < som.fvSize; j++ {
		temp = temp + math.Pow(fv1[j]-fv2[j], 2)
	}
	temp = math.Sqrt(temp)
	return temp
}

// distance calculates distance of two nodes.
func (som SOM) distance(node1 *Node, node2 *Node) float64 {
	return math.Sqrt(math.Pow(float64(node1.x-node2.x), 2) +
		math.Pow(float64(node1.y-node2.y), 2))
}

// basic SOM setup, this shows how SOM can be used.
func BasicRun() {
	fmt.Println("init")

	som := NewSOM(12, 12, 10, 3)
	data := [][]float64{}
	result := [][]float64{}
	test := [][]float64{}

	// training patterns
	data = append(data, []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0})
	result = append(result, []float64{1.0, 0.0, 0.0})
	data = append(data, []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9})
	result = append(result, []float64{0.0, 1.0, 0.0})
	data = append(data, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1})
	result = append(result, []float64{0.0, 0.0, 1.0})

	// testing patterns
	test = append(test, []float64{0.9, 0.8, 0.3, 0.4, 0.4, 0.5, 0.4, 0.3, 0.2, 0.4})
	test = append(test, []float64{0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8})
	test = append(test, []float64{0.1, 0.2, 0.3, 0.4, 0.6, 0.6, 0.4, 0.3, 0.2, 0.1})

	fmt.Println("training")
	som.Train(5000, data, result)

	fmt.Println(data[0], som.predictInt(data[0]))
	fmt.Println(data[1], som.predictInt(data[1]))
	fmt.Println(data[2], som.predictInt(data[2]))

	fmt.Println(test[0], som.predictInt(test[0]))
	fmt.Println(test[1], som.predictInt(test[1]))
	fmt.Println(test[2], som.predictInt(test[2]))

}
