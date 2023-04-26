#pragma once

#include <torch/torch.h>

// common interface of RWKV model, Base class of onnx/libtorch model
class RWKVInterface{
    public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state) = 0;
    torch::Tensor empty_state_;
};