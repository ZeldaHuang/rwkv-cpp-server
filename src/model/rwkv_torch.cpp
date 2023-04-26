#include "rwkv_torch.h"

RWKVTorch::RWKVTorch(int dims, int layers, int headsize)
{
    head = torch::nn::Linear(dims, headsize);
    emb = torch::nn::Embedding(headsize, dims);
    ln_out = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    ln_in = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    for (int i = 0; i < layers; i++)
    {
        blocks.push_back(Block(dims));
    }
    this->eval();
}

RWKVTorch::RWKVTorch(std::string path, c10::ScalarType dtype, c10::ScalarType runtime_dtype)
{
    torch::jit::script::Module w = torch::jit::load(path);
    head = torch::nn::Linear(w.attr("head.weight").toTensor().sizes()[1], w.attr("head.weight").toTensor().sizes()[0]);
    head->weight = w.attr("head.weight").toTensor().to(dtype);
    ln_in = torch::nn::LayerNorm(torch::nn::LayerNormOptions({w.attr("blocks.0.ln0.bias").toTensor().sizes()[0]}));
    ln_in->bias = w.attr("blocks.0.ln0.bias").toTensor().to(runtime_dtype);
    ln_in->weight = w.attr("blocks.0.ln0.weight").toTensor().to(runtime_dtype);
    ln_out = torch::nn::LayerNorm(torch::nn::LayerNormOptions({w.attr("ln_out.weight").toTensor().sizes()[0]}));
    ln_out->weight = w.attr("ln_out.weight").toTensor().to(runtime_dtype);
    ln_out->bias = w.attr("ln_out.bias").toTensor().to(runtime_dtype);
    emb = torch::nn::Embedding(w.attr("emb.weight").toTensor().sizes()[0], w.attr("emb.weight").toTensor().sizes()[1]);
    emb->weight = w.attr("emb.weight").toTensor().to(runtime_dtype);

    for (int i = 0; i < 100; i++)
    {
        if (w.hasattr("blocks." + std::to_string(i) + ".ln1.bias"))
        {
            blocks.push_back(Block(i, w, dtype, runtime_dtype));
        }
        else
        {
            break;
        }
    }
    empty_state_ = w.attr("emptyState").toTensor().to(runtime_dtype);
    dtype_ = dtype;
    this->eval();
}

std::tuple<torch::Tensor, torch::Tensor> RWKVTorch::forward(torch::Tensor x, torch::Tensor state)
{
    bool seq_mode = x.size(0) > 1;
    x = seq_mode ? emb(x) : emb(x[0]);
    x = ln_in(x);
    
    for (int i = 0; i < blocks.size(); i++)
    {
        torch::Tensor rstate;
        std::tie(x, rstate) = seq_mode ? blocks[i].forward_seq(x, state[i]) : blocks[i].forward_one(x, state[i]);
        state[i] = rstate;
    }
    x = seq_mode ? ln_out(x[-1]).to(dtype_) : ln_out(x).to(dtype_);
    torch::Tensor outx = head(x);
    return std::make_tuple(outx, state);
}