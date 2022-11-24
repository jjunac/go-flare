package utils

func MaxFloat64Index(l []float64) (idx int) {
	max := float64(0)
	for i, v := range l {
		if v > max {
			idx = i
			max = v
		}
	}
	return
}
