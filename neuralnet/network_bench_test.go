package neuralnet

import (
	"math/rand"
	"neural-network/utils"
	"testing"
)



func BenchmarkNetworkLearn(b *testing.B) {

	const (
		seed = 711
		nbInputs = 500
		batchSize = 100
	)

	rng := rand.New(rand.NewSource(seed))
	data := utils.InitSlice(nbInputs, func(i int) DataPoint {
		return DataPoint{
			Inputs: utils.InitSlice(1000, func(i int) float64 {
				return rng.Float64()
			}),
			Outputs: utils.InitSlice(50, func(i int) float64 {
				return rng.Float64()
			}),
		}
	})

	network := NewNetwork(
		[]Layer{
			NewLayer(1000, 500, Sigmoid),
			NewLayer(500, 200, Sigmoid),
			NewLayer(200, 50, Sigmoid),
		},
		MSELoss,
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
        network.Learn(*NewDataLoader(data, batchSize, true), 1)
    }

}
