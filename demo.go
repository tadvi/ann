// demo shows how to use various networks included in "ann"
package ann

import (
	"fmt"
)

// RunDemos - call this function to run all the demos.
func RunDemos() {
	SOMPattern()
	BackpropPrimes()

}

// SOMPattern basic SOM setup, this shows how SOM can be used for basic pattern recognition.
func SOMPattern() {
	fmt.Println("-- SOM init")

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

	fmt.Println(data[0], som.PredictInt(data[0]))
	fmt.Println(data[1], som.PredictInt(data[1]))
	fmt.Println(data[2], som.PredictInt(data[2]))

	fmt.Println("running predictions")
	fmt.Println(test[0], som.PredictInt(test[0]))
	fmt.Println(test[1], som.PredictInt(test[1]))
	fmt.Println(test[2], som.PredictInt(test[2]))

}

var primes = []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
	31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
	73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
	127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
	179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
	233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
	283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
	353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
	419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
	467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
	547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
	607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
	661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
	739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
	811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
	877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
	947, 953, 967, 971, 977, 983, 991, 997,
}

// BackpropRun learns prime numbers and then tests if learning worked.
// This should give you about 14 to 28 errors per 1000 tests which is around 2.8 % error rate.
// Your results may slightly vary. Adjust number of training iterations or change number of
// nodes in the hidden layer.
func BackpropPrimes() {
	fmt.Println("-- Backprop init")
	// helper map for checking if it is a prime or not
	checkPrimes := map[int]bool{}
	for _, pr := range primes {
		checkPrimes[pr] = true
	}
	// size of our network input is 2 power of 10 = 1024 and is enough for primes up to 1000
	const inputSize = 10

	// training data is a slice with 1000 prime numbers converted into 1 and 0
	// and outputs are 1 if input is prime number
	tr := []*TrainingData{}

	for i := 0; i < 1000; i++ {
		str := fmt.Sprintf("%010b", i)
		data := &TrainingData{
			Input:  make([]float64, inputSize, inputSize),
			Output: make([]float64, 1, 1),
		}

		// convert inputs into '1' and '0'
		for j := 0; j < inputSize; j++ {
			if str[j] == '1' {
				data.Input[j] = 1
			} else {
				data.Input[j] = 0
			}
		}

		data.Output[0] = 0
		if _, in := checkPrimes[i]; in {
			data.Output[0] = 1
		}

		tr = append(tr, data)
	}

	// input layer has 10 neurons, hidden has 19 and output has 1
	nn := NewBackprop(10, 19, 1)
	fmt.Println("training")
	nn.Train(5000, tr)

	// test if network has been trained	and can recognize prime numbers
	fmt.Println("running predictions")
	errorCount := 0
	for i := 0; i < 1000; i++ {
		testn := i // rand.Intn(1000)
		str := fmt.Sprintf("%010b", testn)
		input := make([]float64, inputSize, inputSize)
		for j := 0; j < inputSize; j++ {
			if str[j] == '1' {
				input[j] = 1
			} else {
				input[j] = 0
			}
		}
		res := nn.PredictInt(input)
		if _, in := checkPrimes[testn]; in {
			if res[0] == 0 {
				errorCount += 1
			}
		} else {
			if res[0] == 1 {
				errorCount += 1
			}
		}
	}

	fmt.Println("total errors", errorCount)

}
