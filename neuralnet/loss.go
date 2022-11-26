package neuralnet

import "encoding/json"

type LossFunc struct {
	Name  string
	Func  func(predicted float64, actual float64) float64
	Prime func(predicted float64, actual float64) float64
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
