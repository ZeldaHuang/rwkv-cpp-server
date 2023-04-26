#pragma once

#include <string>
#include <cwctype>
#include <regex>
#include <iostream>

#include "tokenizer.h"

class RWKVTokenizer
{
public:

    RWKVTokenizer(const std::string &path);

    ~RWKVTokenizer();
    
    std::string decodeTokens(std::vector<uint32_t> &tokens);

    std::vector<uint32_t> encodeTokens(std::string &str);

    Tokenizer_t *tokenizer_ = nullptr;
};
