#pragma once

#include "rwkv_interface.h"
#include "block.h"

// rwkv libtorch model
class RWKVTorch : public torch::nn::Module,public RWKVInterface
{
public:
    RWKVTorch(int dims, int layers, int headsize);

    RWKVTorch(std::string path, c10::ScalarType dtype, c10::ScalarType runtime_dtype, torch::Device device);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state) override;
    
private:
    torch::nn::Linear head = nullptr;
    torch::nn::Embedding emb = nullptr;
    torch::nn::LayerNorm ln_out = nullptr;
    torch::nn::LayerNorm ln_in = nullptr;

    std::vector<Block> blocks = {};
    c10::ScalarType dtype_;
};