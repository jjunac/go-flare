package neuralnet

import (
	"encoding/json"
	"math"
)

type ActivationFunc struct {
	Name   string
	F      func(float64) float64
	FPrime func(float64) float64
}

func (f *ActivationFunc) MarshalJSON() ([]byte, error) {
	return json.Marshal(f.Name)
}

var (
	Sigmoid = ActivationFunc{
		"Sigmoid",
		func(f float64) float64 {
			return 1 / (1 + math.Exp(-f))
		},
		func(f float64) float64 {
			act := 1 / (1 + math.Exp(-f))
			return act * (1 - act)
		},
	}
)
