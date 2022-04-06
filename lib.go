package onnxruntime

// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/cbits -g -O3 -Wno-unused-result
// #cgo CFLAGS: -I${SRCDIR}/cbits -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++17-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo LDFLAGS: -lstdc++
// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include/
// #cgo CFLAGS: -I${SRCDIR}/onnxruntime/include/onnxruntime/core/session/
// #cgo CXXFLAGS: -I${SRCDIR}/onnxruntime/include/
// #cgo CXXFLAGS: -I${SRCDIR}/onnxruntime/include/onnxruntime/core/session/
// #cgo CXXFLAGS: -I${SRCDIR}/onnxruntime/include/onnxruntime/core/common/
// #cgo CXXFLAGS: -I${SRCDIR}/onnxruntime/include/onnxruntime
// #cgo LDFLAGS: -L/usr/lib/ -lonnxruntime
import "C"
