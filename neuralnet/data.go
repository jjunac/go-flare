package neuralnet

import (
	"math/rand"
	"time"
)

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

type DataLoader struct {
	dataset   []DataPoint
	batchSize int
	shuffle   bool
	curr      int
}

func NewDataLoader(dataset []DataPoint, batchSize int, shuffle bool) *DataLoader {
	dl := &DataLoader{
		dataset,
		batchSize,
		shuffle,
		0,
	}
	dl.Reset()
	return dl
}

func (dl *DataLoader) NextBatch() (batch []DataPoint, end bool) {
	lowerBound := dl.curr
	upperBound := dl.curr + dl.batchSize
	if upperBound > len(dl.dataset) {
		upperBound = len(dl.dataset)
		end = true
	}
	dl.curr = upperBound
	batch = dl.dataset[lowerBound:upperBound]
	return
}

func (dl *DataLoader) Reset() {
	dl.curr = 0
	if dl.shuffle {
		rand.Shuffle(len(dl.dataset), func(i, j int) {
			tmp := dl.dataset[i]
			dl.dataset[i] = dl.dataset[j]
			dl.dataset[j] = tmp
		})
	}
}
