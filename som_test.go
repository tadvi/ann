package ann

import (
	"testing"
)

// TestBasic feeds few basic patterns into the SOM, trains it and then
// checks resulting SOM matching probability for similar patterns.
func TestSOMBasic(t *testing.T) {
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

	som.Train(5000, data, result)

	expected := som.PredictInt(test[0])
	if expected[0] < 85 {
		t.Fatal("expected [0] to be above 85% got", expected[0])
	}
	expected = som.PredictInt(test[1])
	if expected[1] < 85 {
		t.Fatal("expected [1] to be above 85% got", expected[1])
	}
	expected = som.PredictInt(test[2])
	if expected[2] < 85 {
		t.Fatal("expected [2] to be above 85% got", expected[2])
	}

}
