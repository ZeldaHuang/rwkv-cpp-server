cd tokenizer
cmake .
cmake --build . --config Release
cd ..
mkdir build
cd build
rmdir /s/q Debug
rmdir /s/q Release
cmake ..
cmake -DCMAKE_PREFIX_PATH="libtorch" ..
mkdir release
mkdir release\assets
cmake --build . --config Release
copy ..\Microsoft.ML.OnnxRuntime.DirectML.1.14.1\runtimes\win-x64\native\onnxruntime.dll .\release\onnxruntime.dll
copy ..\tokenizer\release\tokenizers.dll .\release\tokenizers.dll
xcopy ..\assets\ .\release\assets\  /E