package goflare

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"

	"github.com/jjunac/goflare/utils"
)

func BenchmarkNetworkLearn(b *testing.B) {
	const (
		seed      = 711
		nbInputs  = 2000
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
	)
	optimizer := NewOptimizer(&network, MSELoss, 10, 0)
	loader := NewDataLoader(data, batchSize, true)
	trainer := NetworkTrainer{NbWorkers: 4}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainer.Train(&network, loader, optimizer)
	}
}

func BenchmarkNetworkLearnParallel(b *testing.B) {
	const (
		seed      = 711
		nbInputs  = 2000
		batchSize = 100
	)

	runBench := func(nbWorkers int) func(b *testing.B) {
		return func(b *testing.B) {
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
			)
			optimizer := NewOptimizer(&network, MSELoss, 10, 0)
			loader := NewDataLoader(data, batchSize, true)
			trainer := NetworkTrainer{NbWorkers: nbWorkers}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				trainer.Train(&network, loader, optimizer)
			}
		}
	}

	for _, n := range []int{1, runtime.NumCPU() / 2, runtime.NumCPU(), runtime.NumCPU() * 2} {
		b.Run(fmt.Sprintf("%d_core", n), runBench(n))
	}
}
