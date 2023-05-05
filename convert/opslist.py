import numpy as np


class RWKVOnnxOps():

    def __init__(self, layers, embed, *args, dtype=None, **kwargs):
        import onnx
        self.n_layers = layers
        self.n_embed = embed

        print("embed ", embed)
        
        dtype = onnx.TensorProto.FLOAT if dtype == np.float32 else onnx.TensorProto.FLOAT16 if dtype == np.float16 else onnx.TensorProto.BFLOAT16 if dtype == np.bfloat16 else onnx.TensorProto.FLOAT
        nptype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.float16 if dtype == onnx.TensorProto.FLOAT16 else np.float16 if dtype == onnx.TensorProto.BFLOAT16 else np.float32

        self.nm = 0
        exportname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}.onnx"
        externalname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}.bin"

        # remove old files
        import os
        if os.path.exists(exportname):
            os.remove(exportname)
        if os.path.exists(externalname):
            os.remove(externalname)

        self.TensorList = []
        self.NodeList = []

        def initTensor(x):
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x).astype(nptype)
            else:
                xx = x.squeeze().float().cpu().numpy()
                # convert to float32
                xx = xx.astype(nptype)
            rrx = onnx.helper.make_tensor(
                name,
                dtype,
                xx.shape,
                xx.tobytes(),
                raw=True

            )

            onnx.external_data_helper.set_external_data(
                rrx,
                location=externalname,

            )

            self.TensorList.append(rrx)
            return name

        self.initTensor = initTensor

        def sqrt(x):
            name = f"sqrt_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sqrt',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.sqrt = sqrt

        def mean(x):
            name = f"mean_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.mean = mean

        def relu(x):
            name = f"relu_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Relu',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.relu = relu

        def exp(x):
            name = f"exp_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Exp',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.exp = exp

        def stack(x):
            return [initTensor(r) for r in x]

        self.stack = stack

        def matvec(x, y, is_output=False):
            if is_output:
                name= "output_token"
            else:
                name = f"matvec_{self.nm}_out"
            oname = f"matvec_g_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MatMul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)
            return name

        self.matvec = matvec

        def prod(x):
            name = f"prod_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=[x],
                outputs=[name],
                axes=[1],
                keepdims=0


            )
            self.NodeList.append(node)

            return name

        self.prod = prod

        def mul(x, y):
            name = f"mul_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Mul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.multiply = mul

        def squeeze(x):
            name = f"squeeze_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Squeeze',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        def add(x, y):

            name = f"add_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Add',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.add = add

        def sub(x, y):
            name = f"sub_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sub',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.subtract = sub

        self.one = initTensor([1.0]*embed)

        def lerpx(x, y, z):
            return self.add(x, self.multiply(self.subtract(y, x), z))

        self.lerp = lerpx

        def minimum(x, y):
            name = f"minimum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Min',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.minimum = minimum
        # module def
        self.module = object

        def log(x):
            name = f"log_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Log',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.log = log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def divide(x, y):
            name = f"divide_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Div',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.divide = divide

        def layernorm(x, w, b):
            name = f"layernorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=[x, w, b],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.layernorm = layernorm

        def getIndex(x, y):
            name = f"getIndex_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return squeeze(name)

        self.stackEmbed = False

        def neg(x):
            name = f"neg_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Neg',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.neg = neg

        def logistic(x):
            name = f"logistic_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sigmoid',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.logistical = logistic

        def maximum(x, y):
            name = f"maximum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Max',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.maximum = maximum

        self.getIndex = getIndex

        # convert to float32
        self.emptyState = np.array((([[0.00]*embed, [0.00]*embed, [0.00]*embed, [
            0.00]*embed]+[[-1e30]*embed]))*layers)
        self.emptyState = np.array(self.emptyState, dtype=nptype)

        # self.zero = initTensor([0.0]*embed)
        def reshape(x):
            name = f"output_state"
            self.nm += 1
            node = onnx.helper.make_node(
                'Reshape',
                inputs=[x,"state_shape"],
                outputs=[name],
            )
            self.NodeList.append(node)

            return name
        def concat(x):
            name = f"concat{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Concat',
                inputs=[*x],
                outputs=[name],
                axis=0,
            )

            self.NodeList.append(node)

            return reshape(name)
    
        def getStateIndex(x, y):
            name = f"getStateIndex_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x,y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.getStateIndex= getStateIndex
        self.concat=concat
        self.reshape=reshape
        def ppm(x):
            inputtensor = onnx.helper.make_tensor_value_info("input_token",
                                                             onnx.TensorProto.INT32,
                                                             [1]), "input_token"

            # emptyState = list(map(lambda x: (onnx.helper.make_tensor_value_info("instate"+str(x),
            #                                                                     dtype,
            #                                                                     [embed]), "instate"+str(x)), range(5*layers)))
            emptyState = onnx.helper.make_tensor_value_info("input_state",
                                                             dtype,
                                                             [5*layers,embed]), "input_state"
            print(inputtensor[1],emptyState[1])

            outs = x.forward(
                inputtensor[1], emptyState[1])
            for i in range(5*layers):
                self.TensorList.append(onnx.helper.make_tensor(str(i),onnx.TensorProto.INT32,[],[i]))
            self.TensorList.append(onnx.helper.make_tensor('state_shape',onnx.TensorProto.INT64,[2],[5*layers,embed]))
            print(self.TensorList.__len__())
            print(self.NodeList.__len__())
            print(outs)
            logits = onnx.helper.make_tensor_value_info(outs[0],
                                                        dtype,
                                                        [50277])
            
            state = onnx.helper.make_tensor_value_info(outs[1],
                                                                          dtype,
                                                                          [5*layers,embed])

            # Create the graph (GraphProto)
            graph_def = onnx.helper.make_graph(
                nodes=self.NodeList,  # The list of nodes in the graph.
                name="RWKV",
                # Graph input

                inputs=[inputtensor[0], emptyState[0]],

                outputs=[logits, state],  # Graph output

                initializer=self.TensorList,  # initializer



                # did not work, needs to be external

            )

            modelDef = onnx.helper.make_model(
                graph_def, producer_name="rwkvstic",


            )

            modelDef.opset_import[0].version = 17

            onnx.save(modelDef, exportname)

            # run model
            print("Model saved to: ", exportname, " and is ready to be run")
            print("Data type: ", dtype)
            print("Embedding size: ", embed)
            print("Number of layers: ", layers)
            print("external data: ", externalname)
            exit()
        self.postProcessModule = ppm
