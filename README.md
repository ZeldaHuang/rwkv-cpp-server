
# rwkv-cpp-server
This project enable rwkv model running on windows with C++(**CPU only**).You can run your own rwkv model service without any python dependence(just click a exe file). It provides following features:
- support c tokenizer
- support libtorch and onnxruntime inference
- support server api by [chttplib](https://github.com/yhirose/cpp-httplib)
- provide model convert script to convert rwkv checkpoint to torchscript/onnx file
- provide client and server release file to use from scrath
## Build from source
### Prerequisite
- Visual Studio 2022
- cmake(version>=3.0)
- [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
### Clone the repo
```
git clone --recursive https://github.com/ZeldaHuang/rwkv-cpp-server.git
cd rwkv-cpp-server
```
### Download libtorch
Download libtorch with `curl -O https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.0%2Bcpu.zip` and unzip it to source folder.
### Download onnxruntime
Download [onnxruntime](https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/Microsoft.ML.OnnxRuntime.DirectML.1.14.1.zip) and unzip it to source folder.
### Compile
Run `build.bat`.Release dir path is `build/release`,it contains the `rwkv-server.exe` and all dependence.

## Deploy rwkv server

### Convert models
Download rwkv model from [huggingface](https://huggingface.co/BlinkDL), then convert `.pth` model to torchscript/onnx.
```
python convert/to_onnx.py
python convert/to_torchscript.py
```
Place the torchscript/onnx model in `release/assets/models`. By default the first `.pt` or `.onnx` file in this dir will be loaded.
### Run server
Execute `rwkv-server.exe` in `release` file with `rwkv-server.exe ${model_path} ${ip} ${port}`, you can test the service with `test.py` or open the client app to chat.
