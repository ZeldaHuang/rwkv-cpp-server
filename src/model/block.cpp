#include "block.h"

Block::Block(int dims)
{

    ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    att_key = torch::nn::Linear(dims, dims);
    att_value = torch::nn::Linear(dims, dims);
    att_receptance = torch::nn::Linear(dims, dims);
    att_out = torch::nn::Linear(dims, dims);
    ffn_key = torch::nn::Linear(dims, dims * 4);
    ffn_value = torch::nn::Linear(dims * 4, dims);
    ffn_receptance = torch::nn::Linear(dims, dims);
    time_first = torch::zeros({dims});
    time_decay = torch::zeros({dims});
    att_time_mix_k = torch::zeros({dims});
    att_time_mix_v = torch::zeros({dims});
    att_time_mix_r = torch::zeros({dims});
    ffn_time_mix_k = torch::zeros({dims});
    ffn_time_mix_r = torch::zeros({dims});
}

Block::Block(int i, torch::jit::script::Module w, c10::ScalarType dtype, c10::ScalarType runtime_dtype)
{
    int dims = w.attr("blocks." + std::to_string(i) + ".att.key.weight").toTensor().size(0);
    ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dims}));
    att_key = torch::nn::Linear(dims, dims);
    att_value = torch::nn::Linear(dims, dims);
    att_receptance = torch::nn::Linear(dims, dims);
    att_out = torch::nn::Linear(dims, dims);
    ffn_key = torch::nn::Linear(dims, dims * 4);
    ffn_value = torch::nn::Linear(dims * 4, dims);
    ffn_receptance = torch::nn::Linear(dims, dims);
    time_first = w.attr("blocks." + std::to_string(i) + ".att.time_first").toTensor().squeeze().to(runtime_dtype);
    time_decay = w.attr("blocks." + std::to_string(i) + ".att.time_decay").toTensor().squeeze().exp().neg().to(runtime_dtype);

    att_time_mix_k = w.attr("blocks." + std::to_string(i) + ".att.time_mix_k").toTensor().squeeze().to(runtime_dtype);
    att_time_mix_v = w.attr("blocks." + std::to_string(i) + ".att.time_mix_v").toTensor().squeeze().to(runtime_dtype);
    att_time_mix_r = w.attr("blocks." + std::to_string(i) + ".att.time_mix_r").toTensor().squeeze().to(runtime_dtype);
    ffn_time_mix_k = w.attr("blocks." + std::to_string(i) + ".ffn.time_mix_k").toTensor().squeeze().to(runtime_dtype);
    ffn_time_mix_r = w.attr("blocks." + std::to_string(i) + ".ffn.time_mix_r").toTensor().squeeze().to(runtime_dtype);

    ln1->weight = w.attr("blocks." + std::to_string(i) + ".ln1.weight").toTensor().squeeze().to(runtime_dtype);
    ln1->bias = w.attr("blocks." + std::to_string(i) + ".ln1.bias").toTensor().squeeze().to(runtime_dtype);
    ln2->weight = w.attr("blocks." + std::to_string(i) + ".ln2.weight").toTensor().squeeze().to(runtime_dtype);
    ln2->bias = w.attr("blocks." + std::to_string(i) + ".ln2.bias").toTensor().squeeze().to(runtime_dtype);
    att_key->weight = w.attr("blocks." + std::to_string(i) + ".att.key.weight").toTensor().squeeze().to(dtype).t();
    att_value->weight = w.attr("blocks." + std::to_string(i) + ".att.value.weight").toTensor().squeeze().to(dtype).t();
    att_receptance->weight = w.attr("blocks." + std::to_string(i) + ".att.receptance.weight").toTensor().squeeze().to(dtype).t();
    att_out->weight = w.attr("blocks." + std::to_string(i) + ".att.output.weight").toTensor().squeeze().to(dtype).t();
    ffn_key->weight = w.attr("blocks." + std::to_string(i) + ".ffn.key.weight").toTensor().squeeze().to(dtype).t();
    ffn_value->weight = w.attr("blocks." + std::to_string(i) + ".ffn.value.weight").toTensor().squeeze().to(dtype).t();
    ffn_receptance->weight = w.attr("blocks." + std::to_string(i) + ".ffn.receptance.weight").toTensor().squeeze().to(dtype).t();

    dtype_ = dtype;
    runtime_dtype_ = runtime_dtype;

    block_idx_ = i;
}

torch::Tensor Block::FF_seq(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_r, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw)
{
    auto xx = torch::cat({state[0].to(dtype_).unsqueeze(0), x.slice(0, 0, -1)});
    auto xk = x * time_mix_k + xx * (1 - time_mix_k);
    auto xr = x * time_mix_r + xx * (1 - time_mix_r);
    state[0] = x[-1].to(dtype_);

    auto r = torch::sigmoid(xr.matmul(rw));
    auto k = torch::square(torch::relu(xk.matmul(kw)));
    auto kv = k.matmul(vw);
    return r * kv;
}

torch::Tensor Block::FF_one(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_r, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw)
{
    auto xx = state[0].to(dtype_);
    auto xk = x * time_mix_k + xx * (1 - time_mix_k);
    auto xr = x * time_mix_r + xx * (1 - time_mix_r);
    state[0] = x.to(dtype_);

    auto r = torch::sigmoid(xr.matmul(rw));
    auto k = torch::square(torch::relu(xk.matmul(kw)));
    auto kv = k.matmul(vw);
    return r * kv;
}

torch::Tensor Block::SA_one(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_v, torch::Tensor time_mix_r, torch::Tensor time_first, torch::Tensor time_decay, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw, torch::Tensor ow)
{
    auto xx = state[1].to(dtype_);
    auto xk = x * time_mix_k + xx * (1 - time_mix_k);
    auto xv = x * time_mix_v + xx * (1 - time_mix_v);
    auto xr = x * time_mix_r + xx * (1 - time_mix_r);
    state[1] = x.to(dtype_);

    auto r = torch::sigmoid(xr.matmul(rw));
    auto k = (xk.matmul(kw)).to(dtype_);
    auto v = (xv.matmul(vw)).to(dtype_);

    auto aa = state[2];
    auto bb = state[3];
    auto pp = state[4];
    auto ww = time_first + k;
    auto p = torch::max(pp, ww);
    auto e1 = torch::exp(pp - p);
    auto e2 = torch::exp(ww - p);
    auto a = e1 * aa + e2 * v;
    auto b = e1 * bb + e2;
    ww = pp + time_decay;
    p = torch::max(ww, k);
    e1 = torch::exp(ww - p);
    e2 = torch::exp(k - p);
    state[2] = e1 * aa + e2 * v;
    state[3] = e1 * bb + e2;
    state[4] = p;
    auto wkv = (a / b).to(dtype_);
    return (r * wkv).matmul(ow);
}

torch::Tensor Block::SA_seq(torch::Tensor x, torch::Tensor state, torch::Tensor time_mix_k, torch::Tensor time_mix_v, torch::Tensor time_mix_r, torch::Tensor time_first, torch::Tensor time_decay, torch::Tensor kw, torch::Tensor vw, torch::Tensor rw, torch::Tensor ow)
{
    auto xx = torch::cat({state[1].to(dtype_).unsqueeze(0), x.slice(0, 0, -1)});
    auto xk = x * time_mix_k + xx * (1 - time_mix_k);
    auto xv = x * time_mix_v + xx * (1 - time_mix_v);
    auto xr = x * time_mix_r + xx * (1 - time_mix_r);
    state[1] = x[-1].to(dtype_);

    auto r = torch::sigmoid(xr.matmul(rw));
    auto k = (xk.matmul(kw)).to(dtype_);
    auto v = (xv.matmul(vw)).to(dtype_);

    auto aa = state[2];
    auto bb = state[3];
    auto pp = state[4];
    int T = x.size(0);
    for (int t = 0; t < T; t++)
    {
        auto ww = time_first + k[t];
        auto p = torch::max(pp, ww);
        auto e1 = torch::exp(pp - p);
        auto e2 = torch::exp(ww - p);
        auto a = e1 * aa + e2 * v[t];
        auto b = e1 * bb + e2;
        ww = pp + time_decay;
        p = torch::max(ww, k[t]);
        e1 = torch::exp(ww - p);
        e2 = torch::exp(k[t] - p);
        if (t != T - 1)
        {
            aa = e1 * aa + e2 * v[t];
            bb = e1 * bb + e2;
            pp = p;
        }
        else
        {
            state[2] = e1 * aa + e2 * v[t];
            state[3] = e1 * bb + e2;
            state[4] = p;
        }
        xx[t] = (a / b).to(dtype_);
    }
    return (r * xx).matmul(ow);
}

std::tuple<torch::Tensor, torch::Tensor> Block::forward_seq(torch::Tensor x, torch::Tensor state)
{
    x = x + SA_seq(ln1(x), state, att_time_mix_k, att_time_mix_v, att_time_mix_r, time_first, time_decay, att_key->weight, att_value->weight, att_receptance->weight, att_out->weight);
    x = x + FF_seq(ln2(x), state, ffn_time_mix_k, ffn_time_mix_r, ffn_key->weight, ffn_value->weight, ffn_receptance->weight);

    return std::make_tuple(x, state);
}

std::tuple<torch::Tensor, torch::Tensor> Block::forward_one(torch::Tensor x, torch::Tensor state)
{
    x = x + SA_one(ln1(x), state, att_time_mix_k, att_time_mix_v, att_time_mix_r, time_first, time_decay, att_key->weight, att_value->weight, att_receptance->weight, att_out->weight);
    x = x + FF_one(ln2(x), state, ffn_time_mix_k, ffn_time_mix_r, ffn_key->weight, ffn_value->weight, ffn_receptance->weight);

    return std::make_tuple(x, state);
}
