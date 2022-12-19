package utils

type Set[T comparable] map[T]struct{}

func NewSet[T comparable]() Set[T] {
    return make(Set[T])
}

func NewSetFromSlice[T comparable](keys []T) (s Set[T]) {
    s = make(Set[T], len(keys))
    for _, k := range keys {
        s.Add(k)
    }
    return
}

func (s *Set[T]) Contains(key T) bool {
    _, ok := (*s)[key]
    return ok
}

func (s *Set[T]) Add(key T) {
    (*s)[key] = struct{}{}
}

func (s *Set[T]) Slice() []T {
    values := make([]T, 0, len(*s))
    for k := range *s {
        values = append(values, k)
    }
    return values
}

func (s *Set[T]) Intersection(other Set[T]) Set[T] {
	res := make(Set[T])
	for k := range *s {
		if other.Contains(k) {
			res.Add(k)
		}
	}
	return res
}

func (s *Set[T]) Difference(other Set[T]) Set[T] {
	res := make(Set[T])
	for k := range *s {
		if !other.Contains(k) {
			res.Add(k)
		}
	}
	return res
}
