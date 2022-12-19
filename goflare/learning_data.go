package goflare

import "github.com/jjunac/goflare/utils"

type NetworkLearnData struct {
	Predicted []float64
	Actual    []float64
	LayerData []LayerLearnData
}

func NewNetworkLearnData(n *Network) NetworkLearnData {
	return NetworkLearnData{
		LayerData: utils.InitSlice(len(n.Layers), func(i int) LayerLearnData { return NewLayerLearnData(&n.Layers[i]) }),
		Predicted: make([]float64, 0),
		Actual:    make([]float64, 0),
	}
}

type LayerLearnData struct {
	Inputs         []float64
	WeightedValues []float64
	LossDerivative []float64
}

func NewLayerLearnData(l *Layer) LayerLearnData {
	return LayerLearnData{
		Inputs:         make([]float64, 0),
		WeightedValues: make([]float64, 0),
		LossDerivative: make([]float64, 0),
	}
}
