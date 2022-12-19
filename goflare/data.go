package goflare

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/jjunac/goflare/utils"
)

type Dataset []DataPoint

type DataPoint struct {
	Inputs  []float64
	Outputs []float64
}

type DataStream struct {
	data [][]any
}

func NewDataStream(capacity int) *DataStream {
	return &DataStream{
		make([][]any, 0, capacity),
	}
}

func (ds *DataStream) ApplyPipeline(pipeline *DataPipeline) error {
	return pipeline.Apply(ds.data)
}

// TODO: Find something better to differentiate inputs and targets
func (ds *DataStream) ToDataset(targetCols []int) (dataset Dataset, err error) {
	dataset = make(Dataset, len(ds.data))

	var inputCols []int
	if len(ds.data) > 0 {
		targets := utils.NewSetFromSlice(targetCols)
		inputCols = make([]int, 0, len(ds.data[0])-len(targets))
		for i := range ds.data[0] {
			if !targets.Contains(i) {
				inputCols = append(inputCols, i)
			}
		}
	}

	for row, d := range ds.data {
		columnsToFloat64Slice := func(cols []int) (values []float64, err error) {
			values = make([]float64, len(cols))
			for i, col := range cols {
				if f, ok := d[col].(float64); ok {
					values[i] = f
				} else {
					err = fmt.Errorf("value at row %d, col %d is not a float", row, col)
					return
				}
			}
			return
		}
		dataset[row].Inputs, err = columnsToFloat64Slice(inputCols)
		if err != nil {
			return
		}
		dataset[row].Outputs, err = columnsToFloat64Slice(targetCols)
		if err != nil {
			return
		}
	}
	return
}

type DataLoader struct {
	dataset   Dataset
	batchSize int
	shuffle   bool
}

func NewDataLoader(dataset Dataset, batchSize int, shuffle bool) *DataLoader {
	dl := &DataLoader{
		dataset,
		batchSize,
		shuffle,
	}
	return dl
}

func (dl *DataLoader) Len() int {
	return len(dl.dataset)
}

func (dl *DataLoader) Batches() []Dataset {
	dataset := dl.dataset
	if dl.shuffle {
		dataset = make(Dataset, len(dl.dataset))
		copy(dataset, dl.dataset)
		rand.Shuffle(len(dataset), func(i, j int) {
			tmp := dataset[i]
			dataset[i] = dataset[j]
			dataset[j] = tmp
		})
	}

	batches := make([]Dataset, 0, (len(dataset)+dl.batchSize-1)/dl.batchSize)
	lowerBound := 0
	end := false
	for !end {
		upperBound := lowerBound + dl.batchSize
		if upperBound > len(dataset) {
			upperBound = len(dataset)
			end = true
		}
		batches = append(batches, dataset[lowerBound:upperBound])
		lowerBound = upperBound
	}
	return batches
}

// Little shortcut for RandomSplit with 2 datasets
func RandomSplit2[T any](dataset []T, proportions ...float64) ([]T, []T) {
	res := RandomSplit(dataset, proportions...)
	return res[0], res[1]
}

// Randomly splits a dataset into severals, according to the given proportions.
// [0.7, 0.3], [7, 3], [14, 6] or [70, 30] are all splitting into 2 datasets of resp. 70% and 30%.
func RandomSplit[T any](dataset []T, proportions ...float64) [][]T {
	return RandomSplitWithSource(dataset, proportions, rand.New(rand.NewSource(time.Now().UnixNano())))
}

func RandomSplitWithSource[T any](dataset []T, proportions []float64, rng *rand.Rand) [][]T {
	// Normalize the proportions and transform to cumulative
	sumProportions := float64(0)
	for _, p := range proportions {
		sumProportions += p
	}
	cumul := float64(0)
	for i := range proportions {
		cumul += (proportions[i] / sumProportions)
		proportions[i] = cumul
	}
	// Split into N datasets
	datasets := make([][]T, len(proportions))
	for i := range datasets {
		// Try to estimate the nb of elem (with error rate) to avoid realloc
		datasets[i] = make([]T, 0, int(float64(len(dataset))*proportions[i]*1.05))
	}
	for i := range dataset {
		p := rng.Float64()
		// Find the index of the dataset
		dest := 0
		for ; dest < len(proportions)-1; dest++ {
			if p < proportions[dest] {
				break
			}
		}
		datasets[dest] = append(datasets[dest], dataset[i])
	}
	return datasets
}
