import torch
import torch.nn as nn

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        初始化 LlamaRMSNorm 模块。
        参数:
        - hidden_size: 输入隐藏状态的维度，即需要归一化的特征数。
        - eps: 用于数值稳定的非常小的数，防止计算过程中分母为0（通常为1e-6）。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 权重参数初始化为1
        self.variance_epsilon = eps  # 数值稳定项
 
    def forward(self, hidden_states):
        """
        前向传播函数，执行归一化操作。
        参数:
        - hidden_states: 输入的张量，表示网络层的隐藏状态。
        返回值:
        - 返回归一化并且经过缩放的隐藏状态。
        """
        # 保存输入的原始数据类型，以便最后转换回同样的类型
        input_dtype = hidden_states.dtype
        
        # 计算方差（或更准确地说是每个样本的特征值的均值平方）
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        
        # 对 variance 加上 epsilon，防止分母为0，然后取平方根的倒数，进行归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # weight 是可训练参数，用于缩放归一化后的输出
        return (self.weight * hidden_states).to(input_dtype)