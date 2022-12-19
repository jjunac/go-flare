package goflare

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"

	"github.com/jjunac/goflare/utils"

	"github.com/olekukonko/tablewriter"
)

type ConfusionMatrix struct {
	classes         []string
	confusionMatrix [][]int
	total           int
	totalRight      int
}

func NewConfusionMatrix(classes []string, actual [][]float64, predictions [][]float64) (cm *ConfusionMatrix) {
	cm = &ConfusionMatrix{
		classes:         classes,
		confusionMatrix: utils.InitSlice(len(classes), func(i int) []int { return make([]int, len(classes)) }),
		total:           len(actual),
	}
	for i := range actual {
		actual, _ := utils.Max(actual[i])
		predict, _ := utils.Max(predictions[i])
		cm.confusionMatrix[actual][predict]++
		if actual == predict {
			cm.totalRight++
		}
	}
	return
}

func (cm *ConfusionMatrix) String() string {
	// NOTE: super inneficient, but we don't care. We aren't supposed to print this a zillion times
	sb := strings.Builder{}
	table := tablewriter.NewWriter(&sb)
	table.SetRowLine(true)
	table.SetBorder(false)
	table.Append(append([]string{""}, cm.classes...))
	for i := range cm.classes {
		row := []string{cm.classes[i]}
		for j := range cm.confusionMatrix[i] {
			row = append(row, strconv.Itoa(cm.confusionMatrix[i][j]))
		}
		table.Append(row)
	}

	table.Render()

	lines := make([]string, 0)
	maxLenLine := 0
	scanner := bufio.NewScanner(strings.NewReader(sb.String()))
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) > maxLenLine {
			maxLenLine = len(line)
		}
		lines = append(lines, line)
	}

	sb = strings.Builder{}
	sb.WriteByte('\n')
	paddingActualLen := 11 + maxLenLine/2 - len("Predicted")/2
	for i := 0; i < paddingActualLen; i++ {
		sb.WriteByte(' ')
	}
	sb.WriteString("Predicted\n")
	sb.WriteByte('\n')
	for i := range lines {
		if i == len(lines)/2 {
			sb.WriteString(" Actual    ")
		} else {
			sb.WriteString("           ")
		}
		sb.WriteString(lines[i])
		sb.WriteByte('\n')
	}

	sb.WriteString("\n\n")

	table = tablewriter.NewWriter(&sb)
	table.SetBorder(false)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("  ")
	table.SetRowSeparator("")
	table.SetHeaderLine(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_CENTER)
	table.SetAlignment(tablewriter.ALIGN_RIGHT)
	table.Append([]string{"", "Precision", "Recall", "F1-score", "Support"})
	table.Append([]string{"", "", "", "", ""})

	totalCorrect := 0
	for i := range cm.classes {
		totalCorrect += cm.confusionMatrix[i][i]
	}

	for i := range cm.classes {
		totalActual := utils.Sum(cm.confusionMatrix[i])
		tp := float64(cm.confusionMatrix[i][i])
		fn := float64(totalActual) - tp
		// tn := float64(totalCorrect) - tp
		fp := float64(0)
		for j := range cm.classes {
			fp += float64(cm.confusionMatrix[j][i])
		}
		precision := tp / (tp + fp)
		recall := tp / (tp + fn)
		f1score := 2 * (precision * recall) / (precision + recall)
		row := []string{
			cm.classes[i],
			fmt.Sprintf("%.3f", precision),
			fmt.Sprintf("%.3f", recall),
			fmt.Sprintf("%.3f", f1score),
			strconv.Itoa(totalActual),
		}
		table.Append(row)
	}

	table.Append([]string{"", "", "", "", ""})
	table.Append([]string{"Accuracy", "", "", fmt.Sprintf("%.3f", float64(totalCorrect)/float64(cm.total)), strconv.Itoa(cm.total)})
	table.Render()

	sb.WriteString("\n")

	return sb.String()
}
