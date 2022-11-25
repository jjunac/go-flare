package neuralnet

type LossFunc struct {
	Func  func(actual float64, expected float64) float64
	Prime func(actual float64, expected float64) float64
}

var (
	MSELoss = LossFunc{
		func(actual float64, expected float64) float64 {
			delta := actual - expected
			return delta * delta
		},
		func(actual float64, expected float64) float64 {
			return 2 * (actual - expected)
		},
	}
)
