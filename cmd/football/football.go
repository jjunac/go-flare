package main

import (
	"encoding/csv"
	"flag"
	"io"
	"log"
	"math/rand"
	"neural-network/neuralnet"
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

	f, err := os.Open(".datasets/results_euro2016.csv")
	check(err)

	matches := make([][]string, 0)
	teamsIdx := make(map[string]int)

	r := csv.NewReader(f)
	// date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
	const (
		iDate      = 0
		iHomeTeam  = 1
		iAwayTeam  = 2
		iHomeScore = 3
		iAwayScore = 4
	)
	_, err = r.Read()
	check(err)
	for {
		match, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		matches = append(matches, match)
	}

	for _, match := range matches {
		if _, ok := teamsIdx[match[iHomeTeam]]; !ok {
			teamsIdx[match[iHomeTeam]] = len(teamsIdx)
		}
		if _, ok := teamsIdx[match[iAwayTeam]]; !ok {
			teamsIdx[match[iAwayTeam]] = len(teamsIdx)
		}
	}

	teamsValues := make(map[string]float64, len(teamsIdx))
	for t, i := range teamsIdx {
		teamsValues[t] = float64(i) / float64(len(teamsIdx))
	}

	dataset := make([]neuralnet.DataPoint, 0, len(matches))
	for _, m := range matches {
		homeScore, _ := strconv.Atoi(m[iHomeScore])
		awayScore, _ := strconv.Atoi(m[iAwayScore])
		output := []float64{0, 0, 0}
		if homeScore > awayScore {
			output[0] = 1
		} else if homeScore == awayScore {
			output[1] = 1
		} else {
			output[2] = 1
		}
		dataset = append(dataset, neuralnet.DataPoint{
			Inputs:  []float64{teamsValues[m[iHomeTeam]], teamsValues[m[iAwayTeam]]},
			Outputs: output,
		})
	}

	logrus.Infof("%+v\n", dataset)

	rand.Seed(time.Now().UnixNano())

	network := neuralnet.NewNetwork(
		[]neuralnet.Layer{
			neuralnet.NewLayer(2, 10, neuralnet.Sigmoid),
			neuralnet.NewLayer(10, 3, neuralnet.Sigmoid),
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

		logrus.Infoln(neuralnet.NewConfusionMatrix([]string{"Home win", "tie", "Away win"}, actual, predictions))
	}


	optimizer := neuralnet.NewOptimizer(&network, neuralnet.MSELoss, 0.01, 0)
	trainer := neuralnet.NetworkTrainer{}
	loader := neuralnet.NewDataLoader(trainData, 10, true)

	// debugSvr := tools.NewDebugServer(&network, testData, *neuralnet.NewDataLoader(trainData, 10, true), optimizer)
	// debugSvr.Run("localhost:5000")

	for i := 0; ; i++ {
		runningLoss := trainer.Train(&network, loader, optimizer)
		if i%5000 == 0 {
			logrus.Infof("[%4d] Train data loss = %f\n", i, runningLoss)
			testNetwork(trainData)
			testNetwork(testData)
		}
	}

}
