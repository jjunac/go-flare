package utils

type Number interface {
    int | int64 | float32 | float64
}

func Max[T Number](l []T) (i int, max T) {
	max = 0
	for j, v := range l {
		if v > max {
			i = j
			max = v
		}
	}
	return
}

func Sum[T Number](l []T) (sum T) {
	sum = 0
	for _, v := range l {
		sum += v
	}
	return
}

func InitSlice[T any](size int, compute func(i int)T) []T {
	res := make([]T, size)
	for i := range res {
		res[i] = compute(i)
	}
	return res
}

// Create a zero-ed 2d slice
func MakeSlice2d[T any](rows int, cols int) [][]T {
	res := make([][]T, rows)
	for i := range res {
		res[i] = make([]T, cols)
	}
	return res
}

