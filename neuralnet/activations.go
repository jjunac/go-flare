package neuralnet

import "math"

type ActivationFunc struct {
	Func  func(float64) float64
	Prime func(float64) float64
}

var (
	Sigmoid = ActivationFunc{
		func(f float64) float64 {
			return 1 / (1 + math.Exp(-f))
		},
		func(f float64) float64 {
			act := 1 / (1 + math.Exp(-f))
			return act * (1 - act)
		},
	}
)
