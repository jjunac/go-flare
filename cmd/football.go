package main

import (
	"encoding/csv"
	"flag"
	"io"
	"log"
	"math/rand"
	"neural-network/neuralnet"
	"neural-network/utils"
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

	f, err := os.Open("results_euro2016.csv")
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
	network := neuralnet.NewNetwork(2, 20, 10, 3)

	evaluateWins := func(printMatchGuesses bool) {
		nbRightGuess := 0
		for i := range dataset {
			res := network.Evaluate(dataset[i].Inputs)
			idx := utils.MaxFloat64Index(res)
			answer := "--- WRONG ---"
			if dataset[i].Outputs[idx] > 0.9 {
				answer = "+++ RIGHT +++"
				nbRightGuess++
			}
			if printMatchGuesses {
				logrus.Infof("%s - %s (%s-%s): %s %+v\n",
					matches[i][iHomeTeam],
					matches[i][iAwayTeam],
					matches[i][iHomeScore],
					matches[i][iAwayScore],
					answer,
					res)
			}
		}

		logrus.Infof("##### Final score: %d/%d [%d%%] #####", nbRightGuess, len(dataset), 100*nbRightGuess/len(dataset))
	}

	for i := 0; ; i++ {
		// logrus.Infof("[%d] %+v\n", i, dataset[:1])
		network.Learn(dataset, 0.1)
		if i%1000 == 0 {
			logrus.Infof("[%d] %+v %+v\n", i, network.Evaluate(dataset[0].Inputs), network.Loss(dataset[0]))
			logrus.Infof("[%d] %f\n", i, network.AvgLoss(dataset))
			// logrus.Debugf("[%d] %#v\n", i, network)
			evaluateWins(false)
		}
	}

}
