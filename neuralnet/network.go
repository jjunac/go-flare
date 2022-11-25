package neuralnet

type DataPoint struct {
	Inputs  []float64
	Outputs []float64
}

type Network struct {
	layers []Layer
	loss   LossFunc
}

func NewNetwork(layers []Layer, lossFunc LossFunc) Network {
	return Network{
		layers,
		lossFunc,
	}
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
		loss += n.loss.Func(outputs[i], data.Outputs[i])
	}
	return
}

func (n *Network) AvgLoss(data []DataPoint) (loss float64) {
	for i := range data {
		loss += n.Loss(data[i])
	}
	loss /= float64(len(data))
	return
}

type NetworkLearnData struct {
	LayerData []LayerLearnData
	Predicted []float64
	Actual    []float64
}

func (n *Network) Learn(loader DataLoader, optimizer *Optimizer) {

	nld := NetworkLearnData{
		LayerData: make([]LayerLearnData, len(n.layers)),
	}

	end := false
	var batch []DataPoint
	for !end {
		batch, end = loader.NextBatch()

		for iData := range batch {
			data := &batch[iData]

			// --- Evaluation
			outputs := data.Inputs
			for i := range n.layers {
				nld.LayerData[i].Inputs = outputs
				outputs = n.layers[i].Learn(outputs, &nld.LayerData[i])
			}

			// --- Back-propagation
			nld.Predicted = outputs
			nld.Actual = data.Outputs
			optimizer.Backpropagate(&nld)
		}

		optimizer.Step()
	}

}
