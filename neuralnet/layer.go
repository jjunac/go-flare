package neuralnet

import (
	"math"
	"math/rand"
	"neural-network/utils"
)

type Layer struct {
	NodesIn    int
	NodesOut   int
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunc
}

type LayerLearnData struct {
	Inputs         []float64
	WeightedValues []float64
	LossDerivative []float64
}

func NewLayer(nodesIn int, nodesOut int, activation ActivationFunc) Layer {
	l := Layer{
		nodesIn,
		nodesOut,
		utils.MakeSlice2d[float64](nodesIn, nodesOut),
		make([]float64, nodesOut),
		activation,
	}
	l.Reset()
	return l
}

func (l *Layer) Evaluate(inputs []float64) (outputs []float64) {
	outputs = make([]float64, l.NodesOut)
	for out := range outputs {
		value := l.Biases[out]
		for in := range l.Weights {
			value += inputs[in] * l.Weights[in][out]
		}
		outputs[out] = l.Activation.Func(value)
	}
	return
}

func (l *Layer) Learn(inputs []float64, learnData *LayerLearnData) (outputs []float64) {
	learnData.Inputs = inputs
	learnData.WeightedValues = make([]float64, l.NodesOut)

	outputs = make([]float64, l.NodesOut)
	for out := range outputs {
		value := l.Biases[out]
		for in := range l.Weights {
			value += inputs[in] * l.Weights[in][out]
		}
		learnData.WeightedValues[out] = value
		outputs[out] = l.Activation.Func(value)
	}
	return
}

func (l *Layer) Reset() {
	for out := range l.Biases {
		l.Biases[out] = 0
		for in := range l.Weights {
			l.Weights[in][out] = (rand.Float64()*2 - 1) / math.Sqrt(float64(l.NodesIn))
		}
	}
}
