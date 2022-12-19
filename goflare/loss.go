package goflare

import "encoding/json"

type LossFunc struct {
	Name   string
	F      func(predicted float64, actual float64) float64
	FPrime func(predicted float64, actual float64) float64
}

func (lf *LossFunc) Vectorized(predicted []float64, actual []float64) []float64 {
	res := make([]float64, len(predicted))
	for i := range res {
		res[i] = lf.F(predicted[i], actual[i])
	}
	return res
}

func (lf *LossFunc) PrimeVectorized(predicted []float64, actual []float64) []float64 {
	res := make([]float64, len(predicted))
	for i := range res {
		res[i] = lf.FPrime(predicted[i], actual[i])
	}
	return res
}

func (f *LossFunc) MarshalJSON() ([]byte, error) {
	return json.Marshal(f.Name)
}

var (
	MSELoss = LossFunc{
		"MSE",
		func(predicted float64, actual float64) float64 {
			delta := predicted - actual
			return delta * delta
		},
		func(predicted float64, actual float64) float64 {
			return 2 * (predicted - actual)
		},
	}
)
