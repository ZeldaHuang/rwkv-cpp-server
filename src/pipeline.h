#pragma once

#include "model/rwkv_interface.h"
#include "model/rwkv_onnx.h"
#include "model/rwkv_torch.h"
#include "rwkv_tokenizer.h"

// rwkv pipeline, reference to :https://github.com/BlinkDL/ChatRWKV/blob/db57d70bd151fbbd3fd1fb7e67e18a052cfaab6e/rwkv_pip_package/src/rwkv/utils.py
class RWKVPipeline
{
public:
    RWKVPipeline(std::string &model_path, std::string &tokenizer_path, std::string model_type);
    
    uint32_t sample_logits(torch::Tensor logits, float temperature, float top_p, int top_k);

    std::string generate(std::string &context, float temperature, float top_p, int token_count, float alpha_presence, float alpha_frequency, std::vector<uint32_t> &stop_tokens, std::vector<uint32_t> &ban_tokens);

private:
    std::shared_ptr<RWKVInterface> model_ptr_;
    RWKVTokenizer tokenizer_;
    torch::Tensor prev_state_;
};
