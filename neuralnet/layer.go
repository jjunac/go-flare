package neuralnet

import (
	"math"
	"math/rand"
)

type Layer struct {
	nodesIn    int
	nodesOut   int
	weights    [][]float64
	biases     []float64
	activation ActivationFunc
}

type LayerLearnData struct {
	Inputs         []float64
	WeightedValues []float64
	LossDerivative []float64
}

func NewLayer(nodesIn int, nodesOut int, activation ActivationFunc) Layer {
	return Layer{
		nodesIn,
		nodesOut,
		new2dArray(nodesIn, nodesOut, func(i1, i2 int) float64 { return (rand.Float64()*2 - 1) / math.Sqrt(float64(nodesIn)) }),
		make([]float64, nodesOut),
		activation,
	}
}

// TODO: remove and replace with utils.InitSlice
func new2dArray[T any](rows int, cols int, fn func(int, int) T) [][]T {
	arr := make([][]T, rows)
	for i := range arr {
		arr[i] = make([]T, cols)
		for j := range arr[i] {
			arr[i][j] = fn(i, j)
		}
	}
	return arr
}

func (l *Layer) Evaluate(inputs []float64) (outputs []float64) {
	outputs = make([]float64, l.nodesOut)
	for out := range outputs {
		value := l.biases[out]
		for in := range l.weights {
			value += inputs[in] * l.weights[in][out]
		}
		outputs[out] = l.activation.Func(value)
	}
	return
}

func (l *Layer) Learn(inputs []float64, learnData *LayerLearnData) (outputs []float64) {
	learnData.Inputs = inputs
	learnData.WeightedValues = make([]float64, l.nodesOut)

	outputs = make([]float64, l.nodesOut)
	for out := range outputs {
		value := l.biases[out]
		for in := range l.weights {
			value += inputs[in] * l.weights[in][out]
		}
		learnData.WeightedValues[out] = value
		outputs[out] = l.activation.Func(value)
	}
	return
}
