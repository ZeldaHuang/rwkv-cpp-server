#include "rwkv_onnx.h"

RWKVONNX::RWKVONNX(std::string path): env(ORT_LOGGING_LEVEL_ERROR, "rwkv_onnx"), session_{env, std::wstring(path.begin(), path.end()).c_str(), Ort::SessionOptions{nullptr}}
{

    Ort::TypeInfo state_type_info = session_.GetInputTypeInfo(1);
    state_shape_ = state_type_info.GetTensorTypeAndShapeInfo().GetShape();

    Ort::TypeInfo output_tokens_type_info = session_.GetOutputTypeInfo(0);
    output_tokens_shape_ = output_tokens_type_info.GetTensorTypeAndShapeInfo().GetShape();

    empty_state_ = torch::zeros({state_shape_[0], state_shape_[1]});
    for (int i = 0; i < state_shape_[0] / 5; ++i)
    {
        empty_state_[i * 5 + 3] -= 1e30;
    }
    std::cout << "init onnx model success" << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor> RWKVONNX::forward(torch::Tensor x, torch::Tensor state)
{
    x = x.to(torch::kInt64);
    state = state.to(torch::kFloat32);
    if (true)
    {
        return this->forward_single_token(x, state);
    }
    else
    {
        torch::Tensor input_x = torch::empty({1}).to(torch::kInt64);
        torch::Tensor res_x;
        for (int i = 0; i < x.size(0); ++i)
        {
            input_x[0] = x[i];
            std::tie(res_x, state) = this->forward_single_token(input_x, state);
        }
        return std::make_tuple(res_x, state);
    }
}

std::tuple<torch::Tensor, torch::Tensor> RWKVONNX::forward_single_token(torch::Tensor x, torch::Tensor state)
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tokens_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, x.data<int64_t>(), input_tokens_shape_[0],
                                                                        input_tokens_shape_.data(), input_tokens_shape_.size());
    // onnx input_tensor
    assert(input_tokens_tensor.IsTensor());
    std::vector<Ort::Value> input_tensor;
    input_tensor.push_back(std::move(input_tokens_tensor));

    size_t state_size = state_shape_[0] * state_shape_[1];
    Ort::Value input_state_tensor = Ort::Value::CreateTensor<float>(memory_info, state.data<float>(), state_size,
                                                                        state_shape_.data(), state_shape_.size());
    assert(input_state_tensor.IsTensor());
    input_tensor.push_back(std::move(input_state_tensor));
    // onnx output_tensor

    auto ort_output = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensor.data(), input_tensor.size(), output_names_.data(), output_names_.size());
    // std::cout << "inference done" << std::endl;

    // std::cout<<"onnx inference done "<<output_tensor[0]<<std::endl;
    torch::Tensor output_x = torch::from_blob(ort_output[0].GetTensorMutableData<float>(), {output_tokens_shape_[0]});
    torch::Tensor output_state = torch::from_blob(ort_output[1].GetTensorMutableData<float>(), {state_shape_[0], state_shape_[1]});

    return std::make_tuple(output_x, output_state);
}