#pragma once
#include <iostream>

#include <torch/script.h>
#include <torch/library.h>
#include <torch/torch.h>

// RWKV libtorch Block implementation, reference to pytorch implementation:https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
class Block : public torch::nn::Module
{
public:
    Block(int dims);

    Block(int i, torch::jit::script::Module w, c10::ScalarType dtype, c10::ScalarType runtime_dtype, torch::Device device);

    torch::Tensor FF_seq(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_r, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw);

    torch::Tensor FF_one(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_r, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw);

    torch::Tensor SA_seq(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_v, torch::Tensor time_mix_r, torch::Tensor time_first, torch::Tensor time_decay, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw, torch::Tensor ow);

    torch::Tensor SA_one(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_v, torch::Tensor time_mix_r, torch::Tensor time_first, torch::Tensor time_decay, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw, torch::Tensor ow);

    std::tuple<torch::Tensor, torch::Tensor> forward_seq(torch::Tensor x, torch::Tensor state);

    std::tuple<torch::Tensor, torch::Tensor> forward_one(torch::Tensor x, torch::Tensor state);

private:
    torch::nn::LayerNorm ln1 = nullptr;
    torch::nn::LayerNorm ln2 = nullptr;
    torch::nn::Linear att_key = nullptr;
    torch::nn::Linear att_value = nullptr;
    torch::nn::Linear att_receptance = nullptr;
    torch::nn::Linear att_out = nullptr;
    torch::nn::Linear ffn_key = nullptr;
    torch::nn::Linear ffn_value = nullptr;
    torch::nn::Linear ffn_receptance = nullptr;
    torch::Tensor time_first;
    torch::Tensor time_decay;
    torch::Tensor att_time_mix_k;
    torch::Tensor att_time_mix_v;
    torch::Tensor att_time_mix_r;
    torch::Tensor ffn_time_mix_k;
    torch::Tensor ffn_time_mix_r;
    c10::ScalarType dtype_ = torch::kFloat32;
    c10::ScalarType runtime_dtype_ = torch::kFloat64;

    int block_idx_;
};