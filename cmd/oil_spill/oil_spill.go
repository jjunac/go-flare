package main

import (
	"encoding/csv"
	"flag"
	"io"
	"log"
	"math/rand"
	"neural-network/neuralnet"
	_ "neural-network/tools"
	"os"
	"strconv"
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

	f, err := os.Open(".datasets/oil_spill.csv")
	check(err)
	r := csv.NewReader(f)

	dataset := make([]neuralnet.DataPoint, 0, 1000)

	// Skip header
	_, err = r.Read()
	check(err)
	// Read each rows
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		point := make([]float64, len(row))
		for i := range point {
			f, err := strconv.ParseFloat(row[i], 64)
			check(err)
			point[i] = f
		}

		output := []float64{0, 0}
		if point[49] < 0.5 {
			output[0] = 1
		} else {
			output[1] = 1
		}

		dataset = append(dataset, neuralnet.DataPoint{
			Inputs:  point[:49],
			Outputs: output,
		})
	}

	rand.Seed(time.Now().UnixNano())
	network := neuralnet.NewNetwork(
		[]neuralnet.Layer{
			neuralnet.NewLayer(49, 2, neuralnet.Sigmoid),
		},
	)

	trainData, testData := neuralnet.RandomSplit2(dataset, 7, 3)

	testNetwork := func(data []neuralnet.DataPoint) {
		total := len(data)
		actual := make([][]float64, total)
		predictions := make([][]float64, total)

		for i := range data {
			actual[i] = data[i].Outputs
			predictions[i] = network.Evaluate(data[i].Inputs)
		}

		logrus.Infoln(neuralnet.NewConfusionMatrix([]string{"class0", "class1"}, actual, predictions))
	}

	testNetwork(trainData)
	optimizer := neuralnet.NewOptimizer(&network, neuralnet.MSELoss, 0.1, 0)

	// debugSvr := tools.NewDebugServer(&network, testData, *neuralnet.NewDataLoader(trainData, 100, true), optimizer)
	// debugSvr.Run("localhost:5000")

	trainer := neuralnet.NetworkTrainer{}
	loader := *neuralnet.NewDataLoader(trainData, 100, true)
	for i := 0; i < 100000000; i++ {
		runningLoss := trainer.Train(&network, &loader, optimizer)
		// network.Learn(, optimizer)
		if i%500 == 0 {
			logrus.Infof("[%4d] Train data loss = %f\n", i, runningLoss)
			testNetwork(trainData)
			testNetwork(testData)
		}
	}

}
