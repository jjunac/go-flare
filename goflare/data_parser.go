package goflare

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"

	"github.com/jjunac/goflare/utils"
)

func CSVDataStream(path string) (*DataStream, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open %s: %w", path, err)
	}

	r := csv.NewReader(f)
	ds := NewDataStream(0)

	// Skip header
	_, err = r.Read()
	if err != nil {
		return nil, fmt.Errorf("cannot read csv %s: %w", path, err)
	}

	// Read each rows
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cannot read csv %s: %w", path, err)
		}
		ds.data = append(ds.data, utils.InitSlice(len(row), func(i int) any { return row[i] }))
	}

	return ds, nil
}
