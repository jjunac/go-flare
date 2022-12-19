package matrix

import (
	"math/rand"
	"neural-network/utils"
	"testing"

	"gonum.org/v1/gonum/mat"
)


func BenchmarkVecMatMult(b *testing.B) {
	const (
		nbRow = 500
		nbCol = 500
	)

	b.Run("Slices", func(b *testing.B) {
		vec := utils.InitSlice(nbRow, func(i int) float64 { return rand.Float64() })
		mat := utils.MakeSlice2d[float64](nbRow, nbCol)
		for i := range mat {
			for j := range mat[i] {
				mat[i][j] = rand.Float64()
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := make([]float64, nbCol)
			for row := range mat {
				for col := range mat[row] {
					res[col] += vec[row] * mat[row][col]
				}
			}
		}
	})

	b.Run("Gonum", func(b *testing.B) {
		v := mat.NewDense(1, nbRow, utils.InitSlice(nbRow, func(i int) float64 { return rand.Float64() }))
		m := mat.NewDense(nbRow, nbCol, utils.InitSlice(nbRow * nbCol, func(i int) float64 { return rand.Float64() }))

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := mat.NewDense(1, nbCol, make([]float64, nbCol))
			res.Mul(v, m)
		}
	})

	b.Run("Dense slice", func(b *testing.B) {
		vec := utils.InitSlice(nbRow, func(i int) float64 { return rand.Float64() })
		mat := utils.InitSlice(nbRow * nbCol, func(i int) float64 { return rand.Float64() })

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := make([]float64, nbCol)
			for j := range mat {
				res[j%nbCol] += vec[j%nbCol] * mat[j]
			}
		}
	})
}