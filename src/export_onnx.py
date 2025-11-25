import sys
import torch
import os
import deform_conv2d_onnx_exporter

deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_const

def deform_conv2d_symbolic(g, input, weight, offset, mask, bias,
                           stride_h, stride_w, 
                           padding_h, padding_w,
                           dilation_h, dilation_w,
                           groups,
                           offset_scale,
                           deformable_groups):
    # 使用 _get_const 获取常量值，如果是张量则提取标量
    def get_int_value(x):
        if isinstance(x, int):
            return x
        # 尝试使用 _get_const
        try:
            val = _get_const(x, 'i', 'value')
            return int(val)
        except:
            # 如果是 torch.Tensor，提取值
            if hasattr(x, 'item'):
                return int(x.item())
            return int(x)
    
    stride_h = get_int_value(stride_h)
    stride_w = get_int_value(stride_w)
    padding_h = get_int_value(padding_h)
    padding_w = get_int_value(padding_w)
    dilation_h = get_int_value(dilation_h)
    dilation_w = get_int_value(dilation_w)
    
    return g.op("torchvision::DeformConv2d",
                input, weight, offset, bias, mask,
                stride_i=[stride_h, stride_w],
                padding_i=[padding_h, padding_w],
                dilation_i=[dilation_h, dilation_w])

# 注册到 opset 10
# register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, 10)

EXPORT_ONNX_VERSION = 12

def main(task):
    
    if task == "ctable":
        from engine.ctable.argparser import CTableArgParser
        from engine.ctable.predictor import CTablePredictor
        args = CTableArgParser().parse()
        model = CTablePredictor(args).model
    elif task == "mtable":
        from engine.mtable.argparser import MTableArgParser
        from engine.mtable.predictor import MTablePredictor
        args = MTableArgParser().parse()
        model = MTablePredictor(args).model
    elif task == "stable":
        from engine.stable.argparser import STableArgParser
        from engine.stable.predictor import STablePredictor
        args = STableArgParser().parse()
        model = STablePredictor(args).model
    elif task == "table":
        from engine.table.argparser import TableArgParser
        from engine.table.predictor import TablePredictor
        args = TableArgParser().parse()
        model = TablePredictor(args).model
    else:
        raise ValueError("不支持的 task 类型, 仅支持 ['ctable','mtable','stable','table']")

    model.eval()

    # 2. 创建 dummy input（与训练输入形状一致）
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)  # 示例：batch=1, C=3, H=224, W=224

    # 3. 导出 ONNX
    torch.onnx.export(
        model,                    # 模型
        dummy_input,              # 输入张量（或元组/字典）
        f"model_op{EXPORT_ONNX_VERSION}.onnx",             # 输出文件名
        export_params=True,       # 导出权重
        opset_version=EXPORT_ONNX_VERSION,         # ONNX 算子集版本（推荐 16~18）
        do_constant_folding=True, # 常量折叠优化
        input_names=['input'],    # 输入节点名
        output_names=['output'],  # 输出节点名
        dynamic_axes={            # 动态轴（支持变长输入）
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("ONNX 模型导出成功！")

if __name__ == "__main__":
    # Main function
    main(sys.argv[1] if len(sys.argv) >= 2 and not sys.argv[1].startswith("-") else "mtable")