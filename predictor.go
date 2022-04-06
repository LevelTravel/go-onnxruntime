package onnxruntime

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"gorgonia.org/tensor"
)

type Predictor struct {
	model []byte
	ctx   C.ORT_PredictorContext
}

func New(model []byte) (*Predictor, error) {
	defer PanicOnError()

	modelPtr := unsafe.Pointer(&model[0])

	pred := &Predictor{
		model: model,
		ctx:   C.ORT_NewPredictor(modelPtr, C.size_t(len(model)), C.CPU_DEVICE_KIND, C.int(0)),
	}

	runtime.SetFinalizer(pred, func(p *Predictor) {
		p.Close()
	})

	return pred, GetError()
}

func (p *Predictor) Close() {
	if p == nil {
		return
	}

	if p.ctx != nil {
		C.ORT_PredictorDelete(p.ctx)
	}
	p.ctx = nil
}

func (p *Predictor) Predict(inputs []tensor.Tensor) error {
	defer PanicOnError()
	if len(inputs) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	C.ORT_PredictorClear(p.ctx)

	for _, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return fmt.Errorf("expecting a dense tensor")
		}
		p.addInput(dense)
	}

	C.ORT_PredictorRun(p.ctx)

	return GetError()
}

func (p *Predictor) addInput(ten *tensor.Dense) {
	shape := make([]int64, len(ten.Shape()))
	for i, s := range ten.Shape() {
		shape[i] = int64(s)
	}
	var shapePtr *C.int64_t
	shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	tenPtr := unsafe.Pointer(&ten.Header.Raw[0])

	C.ORT_AddInput(p.ctx, tenPtr, shapePtr, C.int(len(shape)), fromType(ten))

	runtime.KeepAlive(tenPtr)
	runtime.KeepAlive(shape)
}

func (p *Predictor) ReadPredictionOutput() ([]tensor.Tensor, error) {
	defer PanicOnError()

	C.ORT_PredictorConvertOutput(p.ctx)

	cNumOutputs := int(C.ORT_PredictorNumOutputs(p.ctx))

	if cNumOutputs == 0 {
		return nil, fmt.Errorf("zero number of tensors")
	}

	res := make([]tensor.Tensor, cNumOutputs)

	for i := 0; i < cNumOutputs; i++ {
		cPredictions := C.ORT_PredictorGetOutput(p.ctx, C.int(i))
		// The allocated memory will be deleted when destructor of predictor in c++ is called
		res[i] = ortValueToTensor(cPredictions)
	}

	if err := GetError(); err != nil {
		return nil, err
	}

	return res, nil
}
