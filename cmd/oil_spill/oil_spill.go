package main

import (
	"flag"
	"math/rand"
	"time"

	"github.com/jjunac/goflare/goflare"

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

	pipeline := goflare.NewDataPipeline(50, []int{49})
	pipeline.AddRowProcessor(goflare.PPToFloats())
	pipeline.AddInputProcessor(goflare.PPNormalizer())

	data, err := goflare.CSVDataStream(".datasets/oil_spill.csv")
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
	network := goflare.NewNetwork(
		[]goflare.Layer{
			goflare.NewLayer(49, 25, goflare.ReLU),
			goflare.NewLayer(25, 2, goflare.Sigmoid),
		},
	)

	trainData, testData := goflare.RandomSplit2(dataset, 7, 3)

	testNetwork := func(name string, data goflare.Dataset) {
		total := len(data)
		actual := make([][]float64, total)
		predictions := make([][]float64, total)

		for i := range data {
			actual[i] = data[i].Outputs
			predictions[i] = network.Evaluate(data[i].Inputs)
		}

		logrus.Infof("%s data loss = %f\n", name, network.AvgLoss(goflare.MSELoss, data))
		logrus.Infoln(goflare.NewConfusionMatrix([]string{"class0", "class1"}, actual, predictions))
		logrus.Infoln(network.Evaluate(data[0].Inputs), data[0].Outputs)
	}

	// testNetwork(trainData)
	optimizer := goflare.NewOptimizer(&network, goflare.MSELoss, 0.001, 0)

	trainer := goflare.NetworkTrainer{NbWorkers: 6}
	loader := *goflare.NewDataLoader(trainData, len(trainData), true)
	lastLog := time.Now()
	lastEpochLog := 0

	// tools.NewDebugServer(&network, testData, loader, optimizer).Run("localhost:5000")

	for i := 0; i < 100000000; i++ {
		runningLoss := trainer.Train(&network, &loader, optimizer)
		// network.Learn(, optimizer)
		if i%10 == 0 && (time.Since(lastLog) > 2*time.Second) {
			logrus.Infof("#################### Epoch %d [%.1f epoch/s] ####################", i, 1000*float64(i-lastEpochLog)/float64(time.Since(lastLog).Milliseconds()))
			lastLog = time.Now()
			lastEpochLog = i
			logrus.Infof("Train data loss = %f\n", runningLoss)
			testNetwork("Train", trainData)
			testNetwork("Test", testData)
		}
	}

}
