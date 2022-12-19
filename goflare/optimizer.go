package goflare

import (
	"sync"

	"github.com/jjunac/goflare/utils"

	"github.com/sirupsen/logrus"
)

type Optimizer struct {
	nn        *Network
	loss      LossFunc
	d         OptimizerData
	dLock     sync.Mutex
	learnRate float64
	momentum  float64
}

type OptimizerWorker struct {
	nn   Network
	loss LossFunc
	d    OptimizerData
}

type OptimizerData struct {
	layerD []OptimizerLayerData
}

type OptimizerLayerData struct {
	GradientW [][]float64
	VelocityW [][]float64
	GradientB []float64
	VelocityB []float64
}

func NewOptimizer(nn *Network, loss LossFunc, learnRate float64, momentum float64) *Optimizer {
	o := &Optimizer{
		nn:        nn,
		loss:      loss,
		d:         NewOptimizerData(nn),
		learnRate: learnRate,
		momentum:  momentum,
	}
	return o
}

func NewOptimizerData(nn *Network) OptimizerData {
	return OptimizerData{
		layerD: utils.InitSlice(len(nn.Layers), func(i int) OptimizerLayerData {
			l := &nn.Layers[i]
			return OptimizerLayerData{
				GradientW: utils.MakeSlice2d[float64](len(l.Weights), len(l.Weights[0])),
				VelocityW: utils.MakeSlice2d[float64](len(l.Weights), len(l.Weights[0])),
				GradientB: make([]float64, len(l.Biases)),
				VelocityB: make([]float64, len(l.Biases)),
			}
		}),
	}
}

func (self *OptimizerData) Integrate(other *OptimizerData) {
	for i := range other.layerD {
		selfLayerD := self.layerD[i]
		otherLayerD := other.layerD[i]
		// Weight gradients + velocity
		for j := range otherLayerD.GradientW {
			for k := range otherLayerD.GradientW[j] {
				selfLayerD.GradientW[j][k] += otherLayerD.GradientW[j][k]
				selfLayerD.VelocityW[j][k] += otherLayerD.VelocityW[j][k]
			}
		}
		// Bias gradients + velocity
		for j := range otherLayerD.GradientB {
			selfLayerD.GradientB[j] += otherLayerD.GradientB[j]
			selfLayerD.VelocityB[j] += otherLayerD.VelocityB[j]
		}
	}
}

func (o *Optimizer) RunWorker(f func(worker *OptimizerWorker)) {
	// Creating the worker
	w := OptimizerWorker{
		// TODO: Investigate if we really gain perf by copying the network
		CopyNetwork(o.nn),
		o.loss,
		NewOptimizerData(o.nn),
	}
	// Run the worker
	f(&w)
	// Integrate the worker data
	o.dLock.Lock()
	o.d.Integrate(&w.d)
	o.dLock.Unlock()
}

// Backpropgate the errors using the SGD algorithm and stores the gradients internally.
// The network parameters are not updated by this methods, see Optimizer.Step.
func (w *OptimizerWorker) Backpropagate(nld *NetworkLearnData) {
	// --- Last layer handling
	{
		l := &w.nn.Layers[len(w.nn.Layers)-1]
		lld := &nld.LayerData[len(nld.LayerData)-1]
		lld.LossDerivative = make([]float64, len(nld.Predicted))
		for i := range lld.LossDerivative {
			lld.LossDerivative[i] = w.loss.FPrime(nld.Predicted[i], nld.Actual[i])
			logrus.Debugln(lld.LossDerivative[i])
			lld.LossDerivative[i] *= l.Activation.FPrime(lld.WeightedValues[i])
		}

		logrus.Debugf("Actual   : %+v", nld.Actual)
		logrus.Debugf("Predicted: %+v", nld.Predicted)
		logrus.Debugf("Loss'    : %+v", lld.LossDerivative)
		logrus.Debugf("WeigthIn : %+v", lld.WeightedValues)
		logrus.Debugf("Inputs   : %+v", lld.Inputs)
	}

	// --- Propagation from n-1 to 0
	for iLayer := len(w.nn.Layers) - 2; iLayer >= 0; iLayer-- {
		l := &w.nn.Layers[iLayer]
		lld := &nld.LayerData[iLayer]
		lld.LossDerivative = make([]float64, l.NodesOut)
		prev := &w.nn.Layers[iLayer+1]
		prevLld := &nld.LayerData[iLayer+1]

		for node := 0; node < l.NodesOut; node++ {
			valueError := float64(0)
			for i := range prevLld.LossDerivative {
				valueError += prevLld.LossDerivative[i] * prev.Weights[node][i]
			}
			valueError *= l.Activation.FPrime(lld.WeightedValues[node])
			lld.LossDerivative[node] = valueError
		}
	}

	// --- Update gradients
	for iLayer := range w.nn.Layers {
		l := &w.nn.Layers[iLayer]
		lld := &nld.LayerData[iLayer]
		wld := &w.d.layerD[iLayer]
		for nodeOut := 0; nodeOut < l.NodesOut; nodeOut++ {
			lossDerivative := lld.LossDerivative[nodeOut]
			// Update biases
			wld.GradientB[nodeOut] += lossDerivative
			// Update weights
			for nodeIn := 0; nodeIn < l.NodesIn; nodeIn++ {
				wld.GradientW[nodeIn][nodeOut] += lossDerivative * lld.Inputs[nodeIn]
			}
		}
	}
}

// Applies and reset the gradient to the network.
// NOTE: This is *NOT* thread safe
func (o *Optimizer) Step() {
	for iLayer := range o.nn.Layers {
		l := &o.nn.Layers[iLayer]
		ld := o.d.layerD[iLayer]
		weightDecay := float64(1)
		for nodeOut := 0; nodeOut < l.NodesOut; nodeOut++ {
			// Biases
			biasVelocity := ld.VelocityB[nodeOut]*o.momentum - ld.GradientB[nodeOut]*o.learnRate
			ld.VelocityB[nodeOut] = biasVelocity
			l.Biases[nodeOut] += biasVelocity
			ld.GradientB[nodeOut] = 0
			// Weights
			for nodeIn := 0; nodeIn < l.NodesIn; nodeIn++ {
				weight := l.Weights[nodeIn][nodeOut]
				weightVelocity := ld.VelocityW[nodeIn][nodeOut]*o.momentum - ld.GradientW[nodeIn][nodeOut]*o.learnRate
				ld.VelocityW[nodeIn][nodeOut] = weightVelocity
				l.Weights[nodeIn][nodeOut] = weight*weightDecay + weightVelocity
				ld.GradientW[nodeIn][nodeOut] = 0
			}
		}
	}

	o.ZeroGrad()
}

// Reset the internal gradients, typically used at the beginning of a batch
func (o *Optimizer) ZeroGrad() {
	for i := range o.d.layerD {
		ld := &o.d.layerD[i]
		for j := range ld.GradientW {
			for k := range ld.GradientW[j] {
				ld.GradientW[j][k] = 0
			}
		}
		for j := range ld.GradientB {
			ld.GradientB[j] = 0
		}
	}
}
