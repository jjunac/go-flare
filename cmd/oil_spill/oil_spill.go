package main

import (
	"flag"
	"math/rand"
	"neural-network/neuralnet"
	"time"

	"github.com/sirupsen/logrus"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func main() {
	isDebug := flag.Bool("debug", false, "display debug logs")
	flag.Parse()
	if *isDebug {
		logrus.SetLevel(logrus.DebugLevel)
	} else {
		logrus.SetLevel(logrus.InfoLevel)
	}

	pipeline := neuralnet.NewDataPipeline(50, []int{49})
	pipeline.AddRowProcessor(neuralnet.PPToFloats())
	pipeline.AddInputProcessor(neuralnet.PPNormalizer())

	data, err := neuralnet.CSVDataStream(".datasets/oil_spill.csv")
	check(err)
	err = data.ApplyPipeline(pipeline)
	check(err)
	dataset, err := data.ToDataset([]int{49})
	check(err)

	for i := range dataset {
		output := []float64{0, 0}
		if dataset[i].Outputs[0] < 0.5 {
			output[0] = 1
		} else {
			output[1] = 1
		}
		dataset[i].Outputs = output
	}

	rand.Seed(time.Now().UnixNano())
	network := neuralnet.NewNetwork(
		[]neuralnet.Layer{
			neuralnet.NewLayer(49, 25, neuralnet.ReLU),
			neuralnet.NewLayer(25, 2, neuralnet.Sigmoid),
		},
	)

	trainData, testData := neuralnet.RandomSplit2(dataset, 7, 3)

	testNetwork := func(name string, data neuralnet.Dataset) {
		total := len(data)
		actual := make([][]float64, total)
		predictions := make([][]float64, total)

		for i := range data {
			actual[i] = data[i].Outputs
			predictions[i] = network.Evaluate(data[i].Inputs)
		}

		logrus.Infof("%s data loss = %f\n", name, network.AvgLoss(neuralnet.MSELoss, data))
		logrus.Infoln(neuralnet.NewConfusionMatrix([]string{"class0", "class1"}, actual, predictions))
		logrus.Infoln(network.Evaluate(data[0].Inputs), data[0].Outputs)
	}

	// testNetwork(trainData)
	optimizer := neuralnet.NewOptimizer(&network, neuralnet.MSELoss, 0.001, 0)

	trainer := neuralnet.NetworkTrainer{NbWorkers: 6}
	loader := *neuralnet.NewDataLoader(trainData, len(trainData), true)
	lastLog := time.Now()
	lastEpochLog := 0

	// tools.NewDebugServer(&network, testData, loader, optimizer).Run("localhost:5000")

	for i := 0; i < 100000000; i++ {
		runningLoss := trainer.Train(&network, &loader, optimizer)
		// network.Learn(, optimizer)
		if i%10 == 0 && (time.Since(lastLog) > 2 * time.Second) {
			logrus.Infof("#################### Epoch %d [%.1f epoch/s] ####################", i, 1000 * float64(i - lastEpochLog) / float64(time.Since(lastLog).Milliseconds()))
			lastLog = time.Now()
			lastEpochLog = i
			logrus.Infof("Train data loss = %f\n", runningLoss)
			testNetwork("Train", trainData)
			testNetwork("Test", testData)
		}
	}

}
