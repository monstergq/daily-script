import torch, warnings


try:
    import onnxruntime
    onnxruntime_exists = True

except ImportError:
    onnxruntime_exists = False


def run_export(
                model,
                opset: int,
                output: str,
                checkpoint: str,
                gelu_approximate: bool = False,
               ):
    
    """
    将模型导出为 ONNX 格式。

    参数:
        model:实例化后的模型
        checkpoint (str): 模型的检查点路径。
        output (str): ONNX 模型的输出文件路径。
        opset (int): ONNX 操作集版本。
        gelu_approximate (bool): 是否使用近似的 GELU 操作。

    功能:
        - 加载模型。
        - 创建 ONNX 模型，并根据参数设置配置。
        - 如果启用 gelu_approximate，则替换所有 GELU 操作为近似操作。
        - 准备动态轴和虚拟输入用于模型导出。
        - 导出模型到 ONNX 格式。
        - 可选地，使用 ONNXRuntime 验证模型。
    """

    print("Loading model...")
    model.load_state_dict(torch.load(checkpoint))

    # 如果启用了 GELU 近似，则替换 GELU 操作
    if gelu_approximate:

        for _, m in model.named_modules():

            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    # 定义动态轴的配置
    dynamic_axes = {
                        "input": {0: "batch_size", 2: "height", 3: "width"},
                        "output": {0: "batch_size"},
                    }

    # 准备虚拟输入
    batch_size, height, width = 1, 1024, 1024

    dummy_inputs = {
                        'inputs': torch.randn(batch_size, 3, height, width, dtype=torch.float)
                    }

    # 执行一次前向传递以确保模型被正确加载
    _ = model(dummy_inputs['inputs'])

    # 定义 ONNX 模型的输出名称
    output_names = ["s0"]
    # output_names = ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]

    with warnings.catch_warnings():

        # 忽略特定警告
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        with open(output, "wb") as f:

            print(f"Exporting onnx model to {output}...")

            # 导出模型到 ONNX
            torch.onnx.export(
                                model,
                                tuple(dummy_inputs.values()),
                                f,
                                export_params=True,
                                verbose=False,
                                opset_version=opset,
                                do_constant_folding=True,
                                input_names=list(dummy_inputs.keys()),
                                output_names=output_names,
                                dynamic_axes=dynamic_axes,
                             )

    # 如果 ONNXRuntime 可用，则使用它来验证模型
    if onnxruntime_exists:

        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}

        # 设置 CPU 执行提供者
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)

        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()