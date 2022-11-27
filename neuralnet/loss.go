package neuralnet

import "encoding/json"

type LossFunc struct {
	Name  string
	F  func(predicted float64, actual float64) float64
	FPrime func(predicted float64, actual float64) float64
}

func (lf *LossFunc) Vectorized(predicted []float64, actual []float64) (loss float64) {
	for i := range predicted {
		loss += lf.F(predicted[i], actual[i])
	}
	return
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
