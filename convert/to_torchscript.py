from tkinter import filedialog
import torch


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
        dims = my_values["blocks.0.ln0.bias"].shape[0]
        layers = len(list(filter(lambda k: k.startswith(
            "blocks.") and k.endswith(".ln1.bias"), my_values.keys())))

        print("dims", dims)
        print("layers", layers)

        emptyState = torch.zeros(layers, 5, dims)
        for i in range(layers):
            emptyState[i][4] -= 1e30
        setattr(self, "emptyState", emptyState)


# open file  selector, only show  .pth files
path = filedialog.askopenfilename(
    initialdir="./", title="Select file", filetypes=(("pth files", "*.pth"), ("all files", "*.*")))

my_values = torch.load(path, map_location="cpu")

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
output_path = filedialog.asksaveasfilename(
    initialdir="./", title="Select file", filetypes=(("pt files", "*.pt"), ("all files", "*.*")))
container.save(output_path)