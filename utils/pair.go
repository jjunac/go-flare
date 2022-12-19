package utils

type Pair[U any, V any] struct {
    First  U
    Second V
}

func NewPair[U any, V any](first U, second V) Pair[U, V] {
    return Pair[U,V]{first, second}
}


