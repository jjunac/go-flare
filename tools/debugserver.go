package tools

import (
	"encoding/json"
	"net/http"
	"text/template"
	"time"

	"github.com/jjunac/goflare/goflare"

	"github.com/sirupsen/logrus"
)

type H map[string]any

type DebugServer struct {
	nn        *goflare.Network
	trainData []goflare.DataPoint
	testData  []goflare.DataPoint
	epoch     int
}

func NewDebugServer(nn *goflare.Network, testData []goflare.DataPoint, trainLoader goflare.DataLoader, optimizer *goflare.Optimizer) *DebugServer {
	return &DebugServer{
		nn:        nn,
		trainData: testData,
		testData:  testData,
	}
}

func (s *DebugServer) Run(addr string) {
	server := &http.Server{
		Addr:    addr,
		Handler: s.logResquest(s.router()),
	}
	logrus.Infof("Starting debug server: http://%s", addr)
	err := server.ListenAndServe()
	if err != nil && err != http.ErrServerClosed {
		logrus.Fatalf("Could not listen on %s: %v\n", server.Addr, err)
	}
}

func (s *DebugServer) router() *http.ServeMux {
	router := http.NewServeMux()
	router.HandleFunc("/", s.handleRoot)
	router.HandleFunc("/api/network", s.handleApiNetwork)
	router.HandleFunc("/api/learn", s.handleApiLearn)
	router.HandleFunc("/api/reset", s.handleApiReset)
	return router
}

func (s *DebugServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	tmpl, err := template.ParseFiles("tools/debugserverdata/index.html")
	s.checkNoErr(w, err)
	err = tmpl.Execute(w, nil)
	s.checkNoErr(w, err)
}

func (s *DebugServer) handleApiNetwork(w http.ResponseWriter, r *http.Request) {
	err := s.replyJSON(w, 200, s.nn)
	s.checkNoErr(w, err)
}

func (s *DebugServer) handleApiLearn(w http.ResponseWriter, r *http.Request) {
	var query struct {
		Epoch     int
		BatchSize int
		LearnRate float64
	}
	err := s.readJSON(r, &query)
	s.checkNoErr(w, err)

	logrus.Infof("Running training for %d epochs", query.Epoch)
	trainer := goflare.NetworkTrainer{}
	loader := goflare.NewDataLoader(s.trainData, query.BatchSize, true)
	optimizer := goflare.NewOptimizer(s.nn, goflare.MSELoss, query.LearnRate, 0)
	var trainLoss float64
	for i := 0; i < query.Epoch; i++ {
		s.epoch++
		trainLoss = trainer.Train(s.nn, loader, optimizer)
	}

	testLoss := s.nn.AvgLoss(goflare.MSELoss, s.testData)
	logrus.Infof("Test loss: %f", testLoss)

	err = s.replyJSON(w, 200, H{
		"TrainLoss": trainLoss,
		"TestLoss":  testLoss,
		"Epoch":     s.epoch,
	})
	s.checkNoErr(w, err)
}

func (s *DebugServer) handleApiReset(w http.ResponseWriter, r *http.Request) {
	logrus.Infof("Resetting network")
	s.nn.Reset()
	s.epoch = 0
	w.WriteHeader(200)
}

func (s *DebugServer) checkNoErr(w http.ResponseWriter, err error) {
	if err != nil {
		panic(err)
	}
}

func (s *DebugServer) readJSON(r *http.Request, body any) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(body)
}

func (s *DebugServer) replyJSON(w http.ResponseWriter, status int, body any) error {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	return json.NewEncoder(w).Encode(body)
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

func (s *DebugServer) logResquest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lrw := &loggingResponseWriter{w, http.StatusOK}
		logrus.Infof("Processing request %s %s", r.Method, r.URL.Path)
		start := time.Now()
		next.ServeHTTP(lrw, r)
		logrus.Infof("[%d] %s %s (%.3f ms)", lrw.statusCode, r.Method, r.URL.Path, float32(time.Since(start).Microseconds())/1000)
	})
}
