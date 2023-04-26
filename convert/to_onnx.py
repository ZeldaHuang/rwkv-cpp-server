# set these before import RWKV
import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

from rwkv.model import RWKV

from tkinter import filedialog

import torch


def to_onnx(source_path, output_path):
    model = RWKV(model=source_path, strategy='cpu fp32').eval()

    # tokens = torch.tensor([187,187])
    # x = model.w['emb.weight'][tokens]
    state = get_none_state(model.args.n_layer, model.args.n_embd)

    input_names = ['input_token','input_state']
    output_names = ['output_token','output_state']
    torch.onnx.export(model, # 搭建的网络
        (torch.tensor([187]),state), # 输入张量
        output_path,
        input_names=input_names,
        output_names=output_names,
    )
def get_none_state(n_layer, n_embd):
    state = torch.zeros((5*n_layer,n_embd), dtype=torch.float,requires_grad=False,device='cpu')
    for i in range(n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
        state[i*5+3] =- 1e30
    return state

def simple_test(path):
    import onnxruntime as ort
    import numpy as np
    import time
    ort.set_default_logger_severity(3)
    x = np.array([178,178]).astype(np.int64)
    ort_sess = ort.InferenceSession(path)
    inputs= ort_sess.get_inputs()
    for input in inputs:
        print(input.name, input.shape, input.type)
    init_state = get_none_state()
    inputs = {'input_token':x,'input_state':init_state.numpy()}
    t0 = time.time()
    for _ in range(10):
        # for i in range(100):
        out = ort_sess.run(None, inputs)
    t1 = time.time()
    print((t1 - t0)/10)
if __name__ == "__main__":
# open file  selector, only show  .pth files
    path = filedialog.askopenfilename(
    initialdir="./", title="Select file", filetypes=(("pth files", "*.pth"), ("all files", "*.*")))

    # Save arbitrary values supported by TorchScript
    # https://pytorch.org/docs/master/jit.html#supported-type
    output_path = filedialog.asksaveasfilename(
    initialdir="./", title="Select file", filetypes=(("onnx files", "*.onnx"), ("all files", "*.*")))
    to_onnx(path, output_path)
