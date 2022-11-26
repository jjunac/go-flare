package neuralnet

type DataPoint struct {
	Inputs  []float64
	Outputs []float64
}

type Network struct {
	Layers []Layer
	LossFn LossFunc
}

func NewNetwork(layers []Layer, lossFunc LossFunc) Network {
	return Network{
		layers,
		lossFunc,
	}
}

func (n *Network) Evaluate(inputs []float64) []float64 {
	for i := range n.Layers {
		inputs = n.Layers[i].Evaluate(inputs)
	}
	return inputs
}

func (n *Network) Loss(predicted []float64, actual []float64) (loss float64) {
	for i := range predicted {
		loss += n.LossFn.Func(predicted[i], actual[i])
	}
	return
}

func (n *Network) AvgLoss(data []DataPoint) (loss float64) {
	for i := range data {
		loss += n.Loss(n.Evaluate(data[i].Inputs), data[i].Outputs)
	}
	loss /= float64(len(data))
	return
}

type NetworkLearnData struct {
	LayerData []LayerLearnData
	Predicted []float64
	Actual    []float64
}

func (n *Network) Learn(loader DataLoader, optimizer *Optimizer) (runningLoss float64) {

	nld := NetworkLearnData{
		LayerData: make([]LayerLearnData, len(n.Layers)),
	}

	end := false
	var batch []DataPoint
	for !end {
		batch, end = loader.NextBatch()

		for iData := range batch {
			data := &batch[iData]

			// --- Evaluation
			outputs := data.Inputs
			for i := range n.Layers {
				nld.LayerData[i].Inputs = outputs
				outputs = n.Layers[i].Learn(outputs, &nld.LayerData[i])
			}
			runningLoss += n.Loss(outputs, data.Outputs)

			// --- Back-propagation
			nld.Predicted = outputs
			nld.Actual = data.Outputs
			optimizer.Backpropagate(&nld)
		}

		optimizer.Step()
	}

	runningLoss /= float64(loader.batchSize)
	return
}

func (n *Network) Reset() {
	for i := range n.Layers {
		n.Layers[i].Reset()
	}
}
