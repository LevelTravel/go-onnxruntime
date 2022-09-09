# go-onnxruntime

Go binding for Onnxruntime C++ API.
This is [https://github.com/c3sr/go-onnxruntime](https://github.com/c3sr/go-onnxruntime) fork.

## Onnxruntime C++ Library

The binding requires Onnxruntime C++.

The Go binding for Onnxruntime C++ API in this repository is built based on Onnxruntime v1.12.1.

To install Onnxruntime C++ on your system, download runtime libraries from [Onnxruntime](https://github.com/microsoft/onnxruntime).

The Onnxruntime C++ libraries are expected to be under `/usr/lib/`.

If you want to change the paths to the Onnxruntime C++ API, you need to also change the corresponding paths in [lib.go](lib.go) .

## Credits

Some of the logic of conversion between Go types and Ort::Values is borrowed from [go-pytorch](https://github.com/c3sr/go-pytorch).
