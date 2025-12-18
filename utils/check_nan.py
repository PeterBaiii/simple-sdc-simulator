"""
Tensor NaN detection utility with detailed position reporting.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple, List
from .debug import debug

def check_nan(
    tensor: Union[torch.Tensor, np.ndarray],
    name: str = "tensor",
    max_display: int = 10,
    return_positions: bool = False,
    verbose: bool = True
) -> Tuple[bool, List[Tuple]]:
    """
    检测张量中是否存在NaN值，并输出NaN的位置
    
    参数:
        tensor: 要检测的张量 (支持 PyTorch Tensor 或 NumPy array)
        name: 张量的名称，用于输出信息
        max_display: 最多显示的NaN位置数量，None表示显示全部
        return_positions: 是否返回NaN位置列表
        verbose: 是否打印详细信息
    
    返回:
        如果 return_positions=False: 返回是否包含NaN (bool)
        如果 return_positions=True: 返回 (是否包含NaN, NaN位置列表)
    
    示例:
        >>> x = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]])
        >>> check_nan(x, name="x")
        ⚠️  [NaN检测] 张量 'x' 中发现 2 个NaN值！
        张量形状: torch.Size([2, 3])
        NaN位置 (显示前10个):
          #1: (0, 1)
          #2: (1, 2)
    """
    # 转换为统一格式处理
    is_torch = isinstance(tensor, torch.Tensor)
    
    if is_torch:
        # PyTorch张量
        is_nan_mask = torch.isnan(tensor)
        nan_count = torch.sum(is_nan_mask).item()
        device_info = f" [device: {tensor.device}]" if tensor.is_cuda else ""
    else:
        # NumPy数组
        is_nan_mask = np.isnan(tensor)
        nan_count = np.sum(is_nan_mask)
        device_info = ""
    
    has_nan = nan_count > 0
    
    if verbose:
        if has_nan:
            # 获取NaN的位置索引
            if is_torch:
                assert isinstance(is_nan_mask, torch.Tensor)
                nan_indices = torch.nonzero(is_nan_mask, as_tuple=False).tolist()
            else:
                nan_indices = np.argwhere(is_nan_mask).tolist()
            
            # 转换为元组列表
            nan_positions = [tuple(idx) for idx in nan_indices]
            
            # 打印警告信息
            debug(f"\n⚠️  [NaN检测] 张量 '{name}' 中发现 {nan_count} 个NaN值！{device_info}")
            debug(f"张量形状: {tensor.shape}")
            debug(f"张量dtype: {tensor.dtype}")
            
            # 显示NaN位置
            display_count = len(nan_positions) if max_display is None else min(max_display, len(nan_positions))
            debug(f"NaN位置 (显示前{display_count}个):")
            
            for i, pos in enumerate(nan_positions[:display_count]):
                # 格式化位置信息
                if len(pos) == 1:
                    pos_str = f"[{pos[0]}]"
                else:
                    # 使用常见的维度命名
                    dim_names = _get_dimension_names(len(pos))
                    pos_str = f"({', '.join(f'{dim}={idx}' for dim, idx in zip(dim_names, pos))})"
                debug(f"  #{i+1}: {pos_str}")
            
            if len(nan_positions) > display_count:
                debug(f"  ... 还有 {len(nan_positions) - display_count} 个NaN未显示")
            
            debug()  # 空行分隔
            
            if return_positions:
                return has_nan, nan_positions
        else:
            debug(f"✓ [NaN检测] 张量 '{name}' 中没有NaN值{device_info}")
    
    if return_positions:
        if has_nan:
            if is_torch:
                assert isinstance(is_nan_mask, torch.Tensor)
                nan_indices = torch.nonzero(is_nan_mask, as_tuple=False).tolist()
            else:
                nan_indices = np.argwhere(is_nan_mask).tolist()
            nan_positions = [tuple(idx) for idx in nan_indices]
            return has_nan, nan_positions
        else:
            return has_nan, []
    
    return has_nan, []


def _get_dimension_names(ndim: int) -> List[str]:
    """
    根据维度数量返回常用的维度命名
    """
    # 常见的维度命名约定
    common_names = {
        1: ['i'],
        2: ['i', 'j'],
        3: ['b', 'h', 'w'],  # batch, height, width
        4: ['b', 'c', 'h', 'w'],  # batch, channel, height, width
        5: ['b', 'c', 'd', 'h', 'w'],  # batch, channel, depth, height, width
    }
    
    if ndim in common_names:
        return common_names[ndim]
    else:
        # 超过5维使用通用命名
        return [f'd{i}' for i in range(ndim)]


def check_nan_summary(
    tensor: Union[torch.Tensor, np.ndarray],
    name: str = "tensor"
) -> dict:
    """
    返回NaN检测的摘要信息（字典格式）
    
    返回:
        {
            'has_nan': bool,
            'nan_count': int,
            'total_elements': int,
            'nan_ratio': float,
            'shape': tuple,
            'dtype': str
        }
    """
    is_torch = isinstance(tensor, torch.Tensor)
    
    if is_torch:
        is_nan_mask = torch.isnan(tensor)
        nan_count = torch.sum(is_nan_mask).item()
        total_elements = tensor.numel()
    else:
        is_nan_mask = np.isnan(tensor)
        nan_count = np.sum(is_nan_mask)
        total_elements = tensor.size
    
    return {
        'has_nan': nan_count > 0,
        'nan_count': int(nan_count),
        'total_elements': int(total_elements),
        'nan_ratio': float(nan_count) / total_elements if total_elements > 0 else 0.0,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype)
    }


def assert_no_nan(
    tensor: Union[torch.Tensor, np.ndarray],
    name: str = "tensor",
    max_display: int = 5
):
    """
    断言张量中不包含NaN，如果包含则抛出异常
    
    用于调试时的严格检查
    """
    has_nan, positions = check_nan(tensor, name=name, max_display=max_display, 
                                    return_positions=True, verbose=False)
    
    if has_nan:
        pos_str = ", ".join(str(p) for p in positions[:max_display])
        if len(positions) > max_display:
            pos_str += f", ... (共{len(positions)}个)"
        raise ValueError(
            f"张量 '{name}' 中包含 {len(positions)} 个NaN值！\n"
            f"形状: {tensor.shape}\n"
            f"前{max_display}个NaN位置: {pos_str}"
        )
        
# if __name__ == "__main__":
#     debug("=" * 60)
#     debug("NaN检测工具 - 使用示例")
#     debug("=" * 60)
    
#     # 示例1: PyTorch张量 (2D)
#     debug("\n【示例1】2D张量检测:")
#     x = torch.tensor([[1.0, float('nan'), 3.0], 
#                       [4.0, 5.0, float('nan')],
#                       [float('nan'), 8.0, 9.0]])
#     check_nan(x, name="x")
    
#     # 示例2: 3D张量 (batch, height, width)
#     debug("\n【示例2】3D张量检测 (batch, height, width):")
#     y = torch.randn(2, 3, 4)
#     y[0, 1, 2] = float('nan')
#     y[1, 0, 3] = float('nan')
#     y[1, 2, 1] = float('nan')
#     check_nan(y, name="y", max_display=5)
    
#     # 示例3: 4D张量 (batch, channel, height, width)
#     debug("\n【示例3】4D张量检测 (batch, channel, height, width):")
#     z = torch.randn(2, 3, 4, 4)
#     z[0, 1, 2, 3] = float('nan')
#     z[1, 2, 1, 0] = float('nan')
#     check_nan(z, name="z", max_display=10)
    
#     # 示例4: 正常张量（无NaN）
#     debug("\n【示例4】正常张量检测:")
#     normal = torch.randn(3, 4, 5)
#     check_nan(normal, name="normal")
    
#     # 示例5: 大量NaN，限制显示数量
#     debug("\n【示例5】大量NaN检测（限制显示前3个）:")
#     many_nan = torch.randn(5, 5)
#     many_nan[many_nan > 0] = float('nan')  # 将所有正数设为NaN
#     check_nan(many_nan, name="many_nan", max_display=3)
    
#     # 示例6: 获取NaN位置列表
#     debug("\n【示例6】返回NaN位置:")
#     has_nan, positions = check_nan(x, name="x", return_positions=True, verbose=False)
#     debug(f"是否有NaN: {has_nan}")
#     debug(f"NaN位置列表: {positions}")
    
#     # 示例7: 摘要信息
#     debug("\n【示例7】NaN摘要信息:")
#     summary = check_nan_summary(y, name="y")
#     debug(f"摘要: {summary}")
    
#     # 示例8: NumPy数组支持
#     debug("\n【示例8】NumPy数组检测:")
#     np_array = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
#     check_nan(np_array, name="np_array")
    
#     # 示例9: 断言检查（会抛出异常）
#     debug("\n【示例9】断言检查（注释掉以避免异常）:")
#     debug("# assert_no_nan(x, name='x')  # 这会抛出 ValueError")
#     try:
#         assert_no_nan(x, name="x")
#     except ValueError as e:
#         debug(f"捕获到预期的异常:\n{e}")