package neuralnet

import (
	"neural-network/utils"

	"github.com/sirupsen/logrus"
)

type LayerOptimizerParam struct {
	gradientW [][]float64
	velocityW [][]float64
	gradientB []float64
	velocityB []float64
}

type Optimizer struct {
	nn          *Network
	layerParams []LayerOptimizerParam
	learnRate   float64
	momentum    float64
}

func NewOptimizer(nn *Network, learnRate float64, momentum float64) *Optimizer {
	o := &Optimizer{
		nn:        nn,
		learnRate: learnRate,
		momentum:  momentum,
	}
	o.layerParams = utils.InitSlice(len(o.nn.Layers), func(i int) LayerOptimizerParam {
		l := &o.nn.Layers[i]
		return LayerOptimizerParam{
			gradientW: utils.MakeSlice2d[float64](len(l.Weights), len(l.Weights[0])),
			velocityW: utils.MakeSlice2d[float64](len(l.Weights), len(l.Weights[0])),
			gradientB: make([]float64, len(l.Biases)),
			velocityB: make([]float64, len(l.Biases)),
		}
	})
	return o
}

// Reset the internal gradients, typically used at the beginning of a batch
func (o *Optimizer) ZeroGrad() {
	for i := range o.layerParams {
		lp := &o.layerParams[i]
		for j := range lp.gradientW {
			for k := range lp.gradientW[j] {
				lp.gradientW[j][k] = 0
			}
		}
		for j := range lp.gradientB {
			lp.gradientB[j] = 0
		}
	}
}

// Backpropgate the errors using the SGD algorithm and stores the gradients internally.
// The network parameters are not updated by this methods, see Optimizer.Step.
func (o *Optimizer) Backpropagate(nld *NetworkLearnData) {
	var l *Layer
	var lld *LayerLearnData

	// --- Last layer handling
	l = &o.nn.Layers[len(o.nn.Layers)-1]
	lld = &nld.LayerData[len(nld.LayerData)-1]
	lld.LossDerivative = make([]float64, len(nld.Predicted))
	for i := range lld.LossDerivative {
		lld.LossDerivative[i] = o.nn.LossFn.Prime(nld.Predicted[i], nld.Actual[i])
		logrus.Debugln(lld.LossDerivative[i])
		lld.LossDerivative[i] *= l.Activation.Prime(lld.WeightedValues[i])
	}

	// REMOVE
	// lld.LossDerivative[1] *= 10000000
	logrus.Debugf("Actual   : %+v", nld.Actual)
	logrus.Debugf("Predicted: %+v", nld.Predicted)
	logrus.Debugf("Loss'    : %+v", lld.LossDerivative)
	logrus.Debugf("WeigthIn : %+v", lld.WeightedValues)
	logrus.Debugf("Inputs   : %+v", lld.Inputs)

	// --- Propagation from n-1 to 0
	for iLayer := len(o.nn.Layers) - 2; iLayer >= 0; iLayer-- {
		l = &o.nn.Layers[iLayer]
		lld = &nld.LayerData[iLayer]
		lld.LossDerivative = make([]float64, l.NodesOut)
		prev := &o.nn.Layers[iLayer+1]
		prevLld := &nld.LayerData[iLayer+1]

		for node := 0; node < l.NodesOut; node++ {
			valueError := float64(0)
			for i := range prevLld.LossDerivative {
				valueError += prevLld.LossDerivative[i] * prev.Weights[node][i]
			}
			valueError *= l.Activation.Prime(lld.WeightedValues[node])
			lld.LossDerivative[node] = valueError
		}
	}

	// --- Update gradients
	for iLayer := range o.nn.Layers {
		l = &o.nn.Layers[iLayer]
		lld = &nld.LayerData[iLayer]
		lp := o.layerParams[iLayer]
		for nodeOut := 0; nodeOut < l.NodesOut; nodeOut++ {
			lossDerivative := lld.LossDerivative[nodeOut]
			// Update biases
			lp.gradientB[nodeOut] += lossDerivative
			// Update weights
			for nodeIn := 0; nodeIn < l.NodesIn; nodeIn++ {
				// lp.gradientW[nodeIn][nodeOut] += lossDerivative * lld.Inputs[nodeIn]
				lp.gradientW[nodeIn][nodeOut] += lossDerivative
			}
		}
	}
}

// Applies and reset the gradient to the network
func (o *Optimizer) Step() {
	for iLayer := range o.nn.Layers {
		l := &o.nn.Layers[iLayer]
		lp := o.layerParams[iLayer]
		weightDecay := float64(1)
		for nodeOut := 0; nodeOut < l.NodesOut; nodeOut++ {
			// Biases
			biasVelocity := lp.velocityB[nodeOut]*o.momentum - lp.gradientB[nodeOut]*o.learnRate
			lp.velocityB[nodeOut] = biasVelocity
			l.Biases[nodeOut] += biasVelocity
			lp.gradientB[nodeOut] = 0
			// Weights
			for nodeIn := 0; nodeIn < l.NodesIn; nodeIn++ {
				weight := l.Weights[nodeIn][nodeOut]
				weightVelocity := lp.velocityW[nodeIn][nodeOut]*o.momentum - lp.gradientW[nodeIn][nodeOut]*o.learnRate
				lp.velocityW[nodeIn][nodeOut] = weightVelocity
				l.Weights[nodeIn][nodeOut] = weight*weightDecay + weightVelocity
				lp.gradientW[nodeIn][nodeOut] = 0
			}
		}
	}

	o.ZeroGrad()
}
