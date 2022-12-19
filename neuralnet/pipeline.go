package neuralnet

import (
	"errors"
	"fmt"
	"math"
	"neural-network/utils"
	"strconv"
)

type ValueProcessor func(value any) (any, error)
type PipelineProcessor func()ValueProcessor

func TypedValueProcessor[T any](next func(value T) (any, error)) ValueProcessor {
	return func(value any) (any, error) {
		t, ok := value.(T)
		if !ok {
			return nil, fmt.Errorf("not a %T", *new(T))
		}
		return next(t)
	}
}

type DataPipelineOptions func(dp *DataPipeline)

type DataPipeline struct {
	targets utils.Set[int]
	ignored []int
	processors [][]ValueProcessor
}

func NewDataPipeline(nbCols int, targets []int, options ...DataPipelineOptions) *DataPipeline {
	dp := &DataPipeline{
		utils.NewSetFromSlice(targets),
		make([]int, 0),
		utils.MakeSlice2d[ValueProcessor](nbCols, 0),
	}
	for i := range options {
		options[i](dp)
	}
	return dp
}

func IgnoreColumns(ignored []int) DataPipelineOptions {
	return func(dp *DataPipeline) {
		dp.ignored = append(dp.ignored, ignored...)
	}
}

// Applies the previously added values processors.
// NOTE: this mutates the rows passed in param for performance reasons
func (dp *DataPipeline) Apply(rows [][]any) error {
	var err error
	for row := range rows {
		for col := range rows[row] {
			for proc := range dp.processors[col] {
				rows[row][col], err = dp.processors[col][proc](rows[row][col])
				if err != nil {
					return fmt.Errorf("cannot process row %d col %d: %w", row, col, err)
				}
			}
		}
	}
	return nil
}

// Adds a ValueProcessor on specific columns
func (dp *DataPipeline) AddColumnProcessor(cols []int, pp PipelineProcessor) {
	for _, col := range cols {
		dp.processors[col] = append(dp.processors[col], pp())
	}
}

// Adds a ValueProcessor on all the columns
func (dp *DataPipeline) AddRowProcessor(pp PipelineProcessor) {
	for i := range dp.processors {
		dp.processors[i] = append(dp.processors[i], pp())
	}
}

// Adds a ValueProcessor on all the inputs
func (dp *DataPipeline) AddInputProcessor(pp PipelineProcessor) {
	for i := range dp.processors {
		if !dp.targets.Contains(i) {
			dp.processors[i] = append(dp.processors[i], pp())
		}
	}
}

// Adds a ValueProcessor on all the inputs
func (dp *DataPipeline) AddTargetProcessor(pp PipelineProcessor) {
	for i := range dp.processors {
		dp.processors[i] = append(dp.processors[i], pp())
	}
}

func PPStringValueMapper(dictionarySize  int) PipelineProcessor {
	return func() ValueProcessor {
		dict := make(map[string]float64, 0)
		return TypedValueProcessor(func(v string) (any, error) {
			if f, ok := dict[v]; ok {
				// Already in the dict
				return f, nil
			}
			if len(dict) >= dictionarySize {
				return nil, errors.New("dictionary size exceeded")
			}
			f := float64(len(dict)) / float64(dictionarySize)
			dict[v] = f
			return f, nil
		})
	}
}

func PPNormalizer() PipelineProcessor {
	return func() ValueProcessor {
		return TypedValueProcessor(func(v float64) (any, error) {
			return float64(v / (1 + math.Abs(v))), nil
		})
	}
}

func PPToFloats() PipelineProcessor {
	return func() ValueProcessor {
		return func(value any) (any, error) {
			switch v := value.(type) {
			case float64:
				return v, nil
			case float32:
				return float64(v), nil
			case string:
				return strconv.ParseFloat(v, 64)
			default:
				return nil, fmt.Errorf("unsupported type in PPToFloats: %T", v)
			}
		}
	}
}

