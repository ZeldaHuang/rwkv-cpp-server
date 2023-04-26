#include <filesystem>
#include <iostream>
#include <string>
#include "src/rwkv_server.h"

void tokenizer_test()
{
    std::string path = "./assets/20B_tokenizer.json";
    RWKVTokenizer t = RWKVTokenizer("./assets/20B_tokenizer.json");
    std::string s = "\n";
    std::vector<uint32_t> input_tokens = t.encodeTokens(s);
    std::string output_str = t.decodeTokens(input_tokens);
    std::cout << "done: " << output_str << std::endl;
}
void speed_test(RWKVTorch &rwkv)
{
    // warm up
    for (int i = 0; i < 10; i++)
    {
        rwkv.forward(torch::zeros(1).to(torch::kInt32), rwkv.empty_state_.clone());
    }
    std::cout << "finisn warmup" << std::endl;
    torch::Tensor x;
    torch::Tensor state;
    auto time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; ++i)
    {
        std::tie(x, state) = rwkv.forward(torch::ones(100).to(torch::kInt32) * 178, rwkv.empty_state_.clone());
    }
    auto time2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time).count() / 10 << "ms / 100 tokens" << std::endl;
}

void onnx_test()
{
    std::string model_path = "./assets/models/rwkv_model.onnx";

    RWKVONNX onnx_model = RWKVONNX(model_path);
    torch::Tensor out;
    torch::Tensor state;
    for (int i = 0; i < 10; ++i)
    {
        onnx_model.forward(torch::ones(1).to(torch::kInt64) * 178, onnx_model.empty_state_.clone());
    }
    auto time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i)
    {
        std::tie(out, state) = onnx_model.forward(torch::ones(100).to(torch::kInt64), onnx_model.empty_state_.clone());
    }
    auto time2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time).count() / 10 << "ms / 100 tokens" << std::endl;
}

std::string find_model_path(char *argv[])
{
    std::string exe_path = argv[0];
    std::size_t pos = exe_path.find_last_of("\\/");
    exe_path = exe_path.substr(0, pos);
    std::filesystem::path models_assets_path(exe_path + "\\/assets\\/models");
    for (auto &entry : std::filesystem::directory_iterator(models_assets_path))
    {
        std::string model_path = entry.path().string();
        std::size_t pos = model_path.find_last_of(".");
        if (pos >= model_path.size())
        {
            continue;
        }
        if (model_path.substr(pos) == ".pt" || model_path.substr(pos) == ".onnx")
        {
            return model_path;
        }
    }
    return "";
}
void start_server(std::string &model_path, std::string &ip, int port)
{
    std::string tokenizer_path = "./assets/20B_tokenizer.json";
    std::string model_type;
    std::size_t pos = model_path.find_last_of(".");
    if (pos < model_path.size() && model_path.substr(pos) == ".pt")
    {
        model_type = "libtorch";
    }
    else if (pos < model_path.size() && model_path.substr(pos) == ".onnx")
    {
        model_type = "onnx";
    }
    else
    {
        std::cout << "invalid model_path, support xxx.pt or xxx.onnx" << std::endl;
        return;
    }
    std::cout << "using model:" << model_path << std::endl;
    RWKVPipeline pipeline = RWKVPipeline(model_path, tokenizer_path, model_type);
    RWKVServer server = RWKVServer(pipeline);
    server.start(ip, port);
}

int main(int argc, char *argv[])
{
    std::string model_path;
    std::string port;
    std::string ip;

    try
    {
        if (argc <= 1)
        {
            throw "";
        }
        model_path = argv[1];
    }
    catch (...)
    {
        std::cout << "No model_path specified, finding in assets/models" << std::endl;
        model_path = find_model_path(argv);
    }

    try
    {
        if (argc <= 2)
        {
            throw "";
        }
        ip = std::string(argv[2]);
    }
    catch (...)
    {
        std::cout << "No ip specified, default localhost" << std::endl;
        ip = "0.0.0.0";
    }

    try
    {
        if (argc <= 3)
        {
            throw "";
        }
        port = std::string(argv[3]);
    }
    catch (...)
    {
        std::cout << "No port specified, default 5000" << std::endl;
        port = "5000";
    }
    start_server(model_path, ip, std::stoi(port));
    // tokenizer_test();
    // onnx_test();
}