
all: help


## Local dev:

build: ## Build the binary
	go build ./cmd/football.go

run: ## Run the binary
	go run ./cmd/football.go

rund: ## Run the binary with debug logs
	go run ./cmd/football.go -debug


## Profiling of typical usecases:

prof-learn: ## Profile BenchmarkNetworkLearn benchmark
	go test -benchmem -run=^$$ -bench ^BenchmarkNetworkLearn$$ neural-network/neuralnet -cpuprofile learn.prof
	go tool pprof -http=: learn.prof


## Continuous integration:

test: ## Run go test on all modules
	go test ./... -v

bench: ## Run the benchmarks on all modules
	go test ./... -bench=. -run=^#


## Help:

.SILENT: help
help: ## Show this help.
	# Self generating help
	# Inspired from https://gist.github.com/thomaspoignant/5b72d579bd5f311904d973652180c705#file-makefile-L89
	echo 'Usage:'
	echo '  make [target]...'
	echo ''
	echo 'Targets:'
	awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "        %-20s%s\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "\n    %s\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)
	echo ''
