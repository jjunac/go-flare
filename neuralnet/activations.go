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

// Modifies the slice
func (af *ActivationFunc) Vectorized(values []float64) []float64 {
	res := make([]float64, len(values))
	for i := range values {
		res[i] = af.F(values[i])
	}
	return res
}

// Modifies the slice
func (af *ActivationFunc) PrimeVectorized(values []float64) []float64 {
	res := make([]float64, len(values))
	for i := range values {
		res[i] = af.FPrime(values[i])
	}
	return res
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
	ReLU = ActivationFunc{
		"ReLU",
		func(f float64) float64 {
			return math.Max(0, f)
		},
		func(f float64) float64 {
			if f > 0 {
				return 1
			}
			return 0
		},
	}
)
