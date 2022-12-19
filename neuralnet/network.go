package neuralnet

import (
	"neural-network/utils"
)

type Network struct {
	Layers []Layer
}

func NewNetwork(layers []Layer) Network {
	return Network{
		layers,
	}
}

// Returns a new network with the same param as the other, without sharing any object, even slices.
// Useful to avoid sharing when working in parallel
func CopyNetwork(src *Network) Network {
	return Network{
		utils.InitSlice(len(src.Layers), func(i int) Layer { return CopyLayer(&src.Layers[i]) }),
	}
}

func (n *Network) AvgLoss(lossFunc LossFunc, data Dataset) (loss float64) {
	for i := range data {
		loss += utils.Sum(lossFunc.Vectorized(n.Evaluate(data[i].Inputs), data[i].Outputs))
	}
	loss /= float64(len(data))
	return
}

func (n *Network) Evaluate(inputs []float64) []float64 {
	for i := range n.Layers {
		inputs = n.Layers[i].Evaluate(inputs)
	}
	return inputs
}

func (n *Network) EvaluateWithLearnData(inputs []float64, nld *NetworkLearnData) []float64 {
	for i := range n.Layers {
		nld.LayerData[i].Inputs = inputs
		inputs = n.Layers[i].EvaluateWithLearnData(inputs, &nld.LayerData[i])
	}
	return inputs
}


func (n *Network) Reset() {
	for i := range n.Layers {
		n.Layers[i].Reset()
	}
}
