#include "pipeline.h"

RWKVPipeline::RWKVPipeline(std::string &model_path, std::string &tokenizer_path, std::string model_type) : tokenizer_(tokenizer_path)
{
    if(model_type=="libtorch"){
        model_ptr_ = std::make_shared<RWKVTorch>(model_path, torch::kFloat32, torch::kFloat32);
    }
    else{
        model_ptr_ = std::make_shared<RWKVONNX>(model_path);
    }
    prev_state_ = model_ptr_->empty_state_;
}

uint32_t RWKVPipeline::sample_logits(torch::Tensor logits, float temperature=1.0, float top_p=0.85, int top_k=0)
{
    auto probs = torch::softmax(logits, -1);
    auto sorted_ids = torch::argsort(probs);
    auto sorted_probs = probs.index({sorted_ids});
    sorted_probs = torch::flip(sorted_probs, {0});
    auto cumulative_probs = torch::cumsum(sorted_probs, -1);
    // cumulative_probs = cumulative_probs.masked_fill(cumulative_probs < top_p, 0.0);
    auto cutoff = sorted_probs[torch::argmax((cumulative_probs > top_p).to(torch::kInt32))];
    probs = probs.masked_fill(probs < cutoff, 0.0);
    if (top_k < probs.size(0) && top_k > 0) {
        probs.index({sorted_ids.slice(0, -top_k)}) = 0;
    }
    if (temperature != 1.0) {
        probs = torch::pow(probs, 1.0 / temperature);
    }
    auto out = torch::multinomial(probs, 1)[0];
    return (uint32_t)out.item<int>();
}

std::string RWKVPipeline::generate(std::string &context, float temperature, float top_p, int token_count, float alpha_presence, float alpha_frequency, std::vector<uint32_t> &stop_tokens, std::vector<uint32_t> &ban_tokens)
{
    std::vector<uint32_t> input_tokens = tokenizer_.encodeTokens(context);
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor state = prev_state_.clone();
    torch::Tensor last_out;
    std::map<uint32_t, uint32_t> occurrence;
    torch::Tensor t = torch::from_blob(input_tokens.data(), {(int)input_tokens.size()}, opts);
    std::tie(last_out, state) = model_ptr_->forward(t, state);
    std::vector<uint32_t> output_tokens;
    for (int i = 0; i < token_count; ++i)
    {
        for (uint32_t ban_tok : ban_tokens)
        {
            last_out[ban_tok] -=1e30;
        }
        for (auto &par : occurrence)
        {
            uint32_t occurent_tok = par.first;
            uint32_t occurent_cnt = par.second;
            last_out[occurent_tok] -= (alpha_presence + occurent_cnt * alpha_frequency);
        }

        uint32_t tok = sample_logits(last_out, temperature, top_p, 100);
        if (tok == 0 || std::count(stop_tokens.begin(), stop_tokens.end(), tok))
        {
            std::cout<<"break"<<std::endl;
            break;
        }
        occurrence[tok]++;
        output_tokens.push_back(tok);
        t = torch::tensor({(int)tok}).to(torch::kInt32);
        std::tie(last_out, state) = model_ptr_->forward(t, state);
    }
    prev_state_ = state;
    std::string output_str = tokenizer_.decodeTokens(output_tokens);
    
    return output_str;
}