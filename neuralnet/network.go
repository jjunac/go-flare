package neuralnet

type Network struct {
	layers []Layer
}

type DataPoint struct {
	Inputs  []float64
	Outputs []float64
}

func NewNetwork(inputs int, layers ...int) Network {
	n := Network{}
	n.layers = make([]Layer, 0, len(layers))
	prevNbNodes := inputs
	for _, l := range layers {
		n.layers = append(n.layers, NewLayer(prevNbNodes, l))
		prevNbNodes = l
	}
	return n
}

func (n *Network) Evaluate(inputs []float64) []float64 {
	for i := range n.layers {
		inputs = n.layers[i].Evaluate(inputs)
	}
	return inputs
}

func (n *Network) Loss(data DataPoint) (loss float64) {
	outputs := n.Evaluate(data.Inputs)
	for i := range outputs {
		loss += n.lossFunc(outputs[i], data.Outputs[i])
	}
	return
}

func (n *Network) lossFunc(actual float64, expected float64) float64 {
	delta := actual - expected
	return delta * delta
}

func (n *Network) lossFuncPrime(actual float64, expected float64) float64 {
	return 2 * (actual - expected)
}

func (n *Network) AvgLoss(data []DataPoint) (loss float64) {
	for i := range data {
		loss += n.Loss(data[i])
	}
	loss /= float64(len(data))
	return
}

func (n *Network) Learn(dataset []DataPoint, learnRate float64) {

	layerLearnData := make([]LayerLearnData, len(n.layers))

	for iData := range dataset {
		data := &dataset[iData]
	
		// --- Evaluation
		outputs := data.Inputs
		for i := range n.layers {
			layerLearnData[i].Inputs = outputs
			outputs = n.layers[i].Learn(outputs, &layerLearnData[i])
		}

		// --- Back-propagation
		// Last layer handling
		l := &n.layers[len(n.layers)-1]
		lld := &layerLearnData[len(layerLearnData)-1]
		lld.LossDerivative = make([]float64, len(outputs))
		for i := range lld.LossDerivative {
			lld.LossDerivative[i] = n.lossFuncPrime(outputs[i], data.Outputs[i])
			lld.LossDerivative[i] *= l.activationFuncPrime(lld.WeightedValues[i])
		}
		// Propagation from n-1 to 0
		for iLayer := len(n.layers) - 2; iLayer >= 0; iLayer-- {
			l = &n.layers[iLayer]
			lld = &layerLearnData[iLayer]
			lld.LossDerivative = make([]float64, l.nodesOut)
			prev := &n.layers[iLayer+1]
			prevLld := &layerLearnData[iLayer+1]

			for node := 0; node < l.nodesOut; node++ {
				valueError := float64(0)
				for i := range prevLld.LossDerivative {
					valueError += prevLld.LossDerivative[i] * prev.weights[node][i]
				}
				valueError *= l.activationFuncPrime(lld.WeightedValues[node])
				lld.LossDerivative[node] = valueError
			}

		}

		// --- Update layer gradients
		for iLayer := range n.layers {
			l = &n.layers[iLayer]
			lld = &layerLearnData[iLayer]
			for nodeOut := 0; nodeOut < l.nodesOut; nodeOut++ {
				lossDerivative := lld.LossDerivative[nodeOut]
				// Update biases
				l.gradientB[nodeOut] += lossDerivative
				// Update weights
				for nodeIn := 0; nodeIn < l.nodesIn; nodeIn++ {
					l.gradientW[nodeIn][nodeOut] += lossDerivative * lld.Inputs[nodeIn]
				}
			}
		}
	}


	for iLayer := range n.layers {
		l := &n.layers[iLayer]
		l.ApplyGradients(learnRate, 0, 0)
	}
}
