package neuralnet

import (
	"math"
	"math/rand"
)

type Layer struct {
	nodesIn     int
	nodesOut    int
	weights     [][]float64
	gradientW   [][]float64
	velocityW   [][]float64
	biases      []float64
	gradientB   []float64
	velocityB   []float64
}

type LayerLearnData struct {
	Inputs         []float64
	WeightedValues []float64
	LossDerivative []float64
}

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

func NewLayer(nodesIn int, nodesOut int) Layer {
	return Layer{
		nodesIn,
		nodesOut,
		new2dArray(nodesIn, nodesOut, func(i1, i2 int) float64 { return (rand.Float64()*2 - 1) / math.Sqrt(float64(nodesIn)) }),
		new2dArray(nodesIn, nodesOut, func(i1, i2 int) float64 { return 0 }),
		new2dArray(nodesIn, nodesOut, func(i1, i2 int) float64 { return 0 }),
		make([]float64, nodesOut),
		make([]float64, nodesOut),
		make([]float64, nodesOut),
	}
}

func (l *Layer) Evaluate(inputs []float64) (outputs []float64) {
	outputs = make([]float64, l.nodesOut)
	for out := range outputs {
		value := l.biases[out]
		for in := range l.weights {
			value += inputs[in] * l.weights[in][out]
		}
		outputs[out] = l.activationFunc(value)
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
		outputs[out] = l.activationFunc(value)
	}
	return
}

func (l *Layer) ApplyGradients(learnRate float64, regularisation float64, momentum float64) {
	weightDecay := (1 - regularisation * learnRate);
	for nodeOut := 0; nodeOut < l.nodesOut; nodeOut++ {
		// Biases
		biasVelocity := l.velocityB[nodeOut] * momentum - l.gradientB[nodeOut] * learnRate;
		l.velocityB[nodeOut] = biasVelocity;
		l.biases[nodeOut] += biasVelocity;
		l.gradientB[nodeOut] = 0;
		// Weights
		for nodeIn := 0; nodeIn < l.nodesIn; nodeIn++ {
			weight := l.weights[nodeIn][nodeOut];
			weightVelocity := l.velocityW[nodeIn][nodeOut] * momentum - l.gradientW[nodeIn][nodeOut] * learnRate;
			l.velocityW[nodeIn][nodeOut] = weightVelocity;
			l.weights[nodeIn][nodeOut] = weight * weightDecay + weightVelocity;
			l.gradientW[nodeIn][nodeOut] = 0;
		}
	}
}

func (l *Layer) activationFunc(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func (l *Layer) activationFuncPrime(value float64) float64 {
	act := l.activationFunc(value)
	return act * (1-act)
}
