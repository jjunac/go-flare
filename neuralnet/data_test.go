package neuralnet

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRandomSplitWithSource(t *testing.T) {
	assert := assert.New(t)
	type args struct {
		dataset     []int
		proportions []float64
	}
	tests := []struct {
		name      string
		args      args
		exprected [][]int
	}{
		{
			"Basic with decimal proportions",
			args{[]int{1, 2, 3, 4, 5}, []float64{0.4, 0.6}},
			[][]int{{2}, {1, 3, 4, 5}},
		},
		{
			"Basic with percentage proportions",
			args{[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{20, 40, 10, 10}},
			[][]int{{}, {1, 2, 3, 6, 7}, {8}, {4, 5, 9, 10}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(1337))
			assert.Equal(tt.exprected, RandomSplitWithSource(tt.args.dataset, tt.args.proportions, rng))
		})
	}

	// Testing on a huge number to watch the distribution
	t.Run("Distribution with 10000 elements", func(t *testing.T) {
		numbers := make([]int, 10000)
		for i := range numbers {
			numbers[i] = i
		}
		rng := rand.New(rand.NewSource(1337))
		datasets := RandomSplitWithSource(numbers, []float64{20, 40, 10, 10}, rng)
		lenghts := make([]int, 4)
		for i := range datasets {
			lenghts[i] = len(datasets[i])
		}
		assert.Equal([]int{2435, 4998, 1295, 1272}, lenghts)
	})
}
