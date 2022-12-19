package goflare

import (
	"runtime"
	"sync"

	"github.com/jjunac/goflare/utils"
)

type NetworkTrainer struct {
	NbWorkers int
}

func (nt *NetworkTrainer) nbWorkers() int {
	if nt.NbWorkers > 0 {
		return nt.NbWorkers
	}
	// Don't exactly know why, but NumCPU actually make the program slower.
	// It could be io bound, cache miss because of sharing/false-sharing, hyper threading, ...
	// TODO: Investigate
	return runtime.NumCPU() / 2
}

func (nt *NetworkTrainer) Train(n *Network, loader *DataLoader, optimizer *Optimizer) (globalRunningLoss float64) {
	for _, batch := range loader.Batches() {
		type learningResult struct {
			runningLoss float64
		}
		dataPointChannel := make(chan *DataPoint, nt.nbWorkers())
		resultChannel := make(chan *learningResult, nt.nbWorkers())
		var workerWg sync.WaitGroup

		for i := 0; i < nt.nbWorkers(); i++ {
			workerWg.Add(1)
			go optimizer.RunWorker(func(worker *OptimizerWorker) {
				defer workerWg.Done()
				res := learningResult{}
				nld := NewNetworkLearnData(n)
				for {
					data, open := <-dataPointChannel
					if !open {
						break
					}

					// --- Evaluation
					outputs := n.EvaluateWithLearnData(data.Inputs, &nld)
					res.runningLoss += utils.Sum(optimizer.loss.Vectorized(outputs, data.Outputs))

					// --- Back-propagation
					nld.Predicted = outputs
					nld.Actual = data.Outputs
					worker.Backpropagate(&nld)
				}
				resultChannel <- &res
			})
		}

		for iData := range batch {
			dataPointChannel <- &batch[iData]
		}
		close(dataPointChannel)

		workerWg.Wait()

		close(resultChannel)
		for res := range resultChannel {
			globalRunningLoss += res.runningLoss
		}

		optimizer.Step()
	}

	globalRunningLoss /= float64(loader.batchSize)
	return
}
