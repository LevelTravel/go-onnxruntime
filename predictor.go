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

func init() {
	C.ORT_Init()
}

type Predictor struct {
	model []byte
	ctx   C.ORT_PredictorContext
}

func New(model []byte) (*Predictor, error) {
	modelPtr := unsafe.Pointer(&model[0])

	ret := C.ORT_NewPredictor(modelPtr, C.size_t(len(model)), C.CPU_DEVICE_KIND, C.int(0))
	if ret.pstrErr != nil {
		s := C.GoString(ret.pstrErr)
		C.free(unsafe.Pointer(ret.pstrErr))
		return nil, fmt.Errorf(s)
	}

	pred := &Predictor{
		model: model,
		ctx:   ret.ctx,
	}

	runtime.SetFinalizer(pred, func(p *Predictor) {
		p.Close()
	})

	return pred, nil
}

func (p *Predictor) Close() {
	if p == nil {
		return
	}

	if p.ctx != nil {
		C.ORT_PredictorDelete(p.ctx)
		p.ctx = nil
		p.model = nil
	}
}

func (p *Predictor) Predict(inputs []tensor.Tensor) error {
	if len(inputs) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	errMsg := C.ORT_PredictorClear(p.ctx)
	if errMsg != nil {
		s := C.GoString(errMsg)
		C.free(unsafe.Pointer(errMsg))
		return fmt.Errorf(s)
	}

	for _, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return fmt.Errorf("expecting a dense tensor")
		}
		if err := p.addInput(dense); err != nil {
			return err
		}
	}

	errMsg = C.ORT_PredictorRun(p.ctx)
	if errMsg != nil {
		s := C.GoString(errMsg)
		C.free(unsafe.Pointer(errMsg))
		return fmt.Errorf(s)
	}

	return nil
}

func (p *Predictor) addInput(ten *tensor.Dense) error {
	shape := make([]int64, len(ten.Shape()))
	for i, s := range ten.Shape() {
		shape[i] = int64(s)
	}
	var shapePtr *C.int64_t
	shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	tenPtr := unsafe.Pointer(&ten.Header.Raw[0])

	errMsg := C.ORT_AddInput(p.ctx, tenPtr, shapePtr, C.int(len(shape)), fromType(ten))
	if errMsg != nil {
		s := C.GoString(errMsg)
		C.free(unsafe.Pointer(errMsg))
		return fmt.Errorf(s)
	}

	runtime.KeepAlive(tenPtr)
	runtime.KeepAlive(shape)

	return nil
}

func (p *Predictor) ReadPredictionOutput() ([]tensor.Tensor, error) {
	errMsg := C.ORT_PredictorConvertOutput(p.ctx)
	if errMsg != nil {
		s := C.GoString(errMsg)
		C.free(unsafe.Pointer(errMsg))
		return nil, fmt.Errorf(s)
	}

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

	return res, nil
}
