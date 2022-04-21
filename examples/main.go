package main

import (
	"fmt"
	"log"
	"os"

	"github.com/LevelTravel/go-onnxruntime"
	"gorgonia.org/tensor"
)

func main() {
	model, err := os.ReadFile("./examples/model.onnx")
	if err != nil {
		log.Fatalf("model load failed %v", err)
	}

	predictor, err := onnxruntime.New(model)
	if err != nil {
		log.Fatalf("Onnxruntime predictor initialization failed: %v", err)
	}
	defer predictor.Close()

	err = predictor.Predict([]tensor.Tensor{
		tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
			tensor.WithShape(3, 4),
		),
		tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithBacking([]float32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}),
			tensor.WithShape(4, 3),
		),
	})
	if err != nil {
		log.Fatalf("Onnxruntime predictor predicting failed: %v", err)
	}

	output, err := predictor.ReadPredictionOutput()
	if err != nil {
		log.Fatalf("Onnxruntime predictor read prediction output failed: %v", err)
	}

	fmt.Println(output[0].Data().([]float32))
}
