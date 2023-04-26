#include "rwkv_tokenizer.h"

RWKVTokenizer::RWKVTokenizer(const std::string &path)
{
    tokenizer_ = tokenizer_create_from_file(path.c_str());
}

RWKVTokenizer::~RWKVTokenizer()
{
    if (tokenizer_ != nullptr)
    {
        tokenizer_destroy(tokenizer_);
    }
}

std::string RWKVTokenizer::decodeTokens(std::vector<uint32_t> &tokens)
{
    auto allocator = [](size_t size, void *payload) -> void *
    {
        return malloc(size);
    };
    CArrayRef<uint32_t> input_ids = CArrayRef<uint32_t>(tokens);
    c_str_t c_str;
    tokenizer_decode(tokenizer_, true, 0, allocator, &input_ids, &c_str);
    std::string decode_str = c_str.to_string();
    free((void *)c_str.ptr);
    return decode_str;
}

std::vector<uint32_t> RWKVTokenizer::encodeTokens(std::string &str)
{
    Encoding_t *encoded = tokenizer_encode(tokenizer_, str.c_str(), true);
    CArrayRef<uint32_t> array;
    std::vector<uint32_t> tokens_vec;
    encoding_get_ids(encoded, &array);
    for (int i = 0; i < array.length; i++)
    {
        tokens_vec.push_back(array[i]);
    }
    return tokens_vec;
}