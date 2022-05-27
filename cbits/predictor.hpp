#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#include <stdbool.h>
#ifdef __cplusplus
#include <onnxruntime_cxx_api.h>
extern "C" {
#else
#include <onnxruntime_c_api.h>
#endif  /* __cplusplus */

    typedef struct ORT_Value {
        ONNXTensorElementDataType otype;
        void *data_ptr;
        int64_t *shape_ptr;
        size_t shape_len;
    } ORT_Value;

    typedef enum { UNKNOWN_DEVICE_KIND = -1, CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } ORT_DeviceKind;
    typedef void* ORT_PredictorContext;

    typedef struct NP_HANDLE_ERR {
        ORT_PredictorContext ctx;
        const char *pstrErr;
    } NP_HANDLE_ERR;

    // Predictor interface for Go

    NP_HANDLE_ERR ORT_NewPredictor(const char *model_file);

    void ORT_Init();

    char* ORT_PredictorClear(ORT_PredictorContext pred);

    char* ORT_PredictorRun(ORT_PredictorContext pred);

    char* ORT_PredictorConvertOutput(ORT_PredictorContext pred);

    int ORT_PredictorNumOutputs(ORT_PredictorContext pred);

    ORT_Value ORT_PredictorGetOutput(ORT_PredictorContext pred, int index);

    void ORT_PredictorDelete(ORT_PredictorContext pred);

    char* ORT_AddInput(ORT_PredictorContext pred, void *input, int64_t *dimensions,
                    int n_dim, ONNXTensorElementDataType dtype);

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif /* __PREDICTOR_HPP__ */
