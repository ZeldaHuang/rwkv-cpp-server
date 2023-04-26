#pragma once

#include <vector>
#include <iostream>
#include <cassert>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include "rwkv_interface.h"

//rwkv onnx model
class RWKVONNX : public RWKVInterface
{
public:
    RWKVONNX(std::string path);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state);

    std::tuple<torch::Tensor, torch::Tensor> forward_single_token(torch::Tensor x, torch::Tensor state);

private:
    Ort::Env env;
    Ort::Session session_;
    std::vector<int64_t> input_tokens_shape_ = {1};
    std::vector<int64_t> output_tokens_shape_;
    std::vector<int64_t> state_shape_;
    std::vector<const char *> input_names_ = {"input_token", "input_state"};
    std::vector<const char *> output_names_ = {"output_token", "output_state"};
};
