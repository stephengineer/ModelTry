import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 模型总维度（如512）
        self.num_heads = num_heads  # 头数（如8）
        self.d_k = d_model // num_heads  # 每个头的维度（如512/8=64）

        # 定义Q、K、V的投影层和输出层
        self.w_q = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 整合多头输出

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力（单头）"""
        # q: (batch_size, num_heads, seq_len_q, d_k)
        # k: (batch_size, num_heads, seq_len_k, d_k)
        # v: (batch_size, num_heads, seq_len_v, d_v) （通常seq_len_k = seq_len_v）

        # 计算注意力分数：Q·K^T / sqrt(d_k)：(batch_size, num_heads, seq_len_q, seq_len_k)
        import math

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask（例如：遮挡padding或未来信息）
        if mask is not None:
            # mask形状需与scores匹配，通常为(batch_size, 1, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重（softmax归一化）
        attn_weights = torch.softmax(
            scores, dim=-1
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # 加权求和V
        output = torch.matmul(
            attn_weights, v
        )  # (batch_size, num_heads, seq_len_q, d_v)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        # 输入形状：q, k, v -> (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # 1. 投影并拆分多头: (batch_size, seq_len_q, d_model) -> (batch_size, num_heads, seq_len_q, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力
        output, attn_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )  # output: (batch_size, num_heads, seq_len_q, d_k)

        # 3. 拼接多头输出
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        # 4. 输出层整合
        output = self.w_o(output)  # (batch_size, seq_len_q, d_model)
        return output, attn_weights


def test_multi_head_attention():
    # 配置参数
    batch_size = 2  # 批次大小
    seq_len = 5  # 序列长度（如5个词）
    d_model = 128  # 模型总维度
    num_heads = 8  # 头数（128/8=16，每个头维度16）

    # 1. 创建随机输入（模拟词嵌入）
    q = torch.randn(batch_size, seq_len, d_model)  # 随机查询
    k = torch.randn(batch_size, seq_len, d_model)  # 随机键（与查询同序列长度）
    v = torch.randn(batch_size, seq_len, d_model)  # 随机值（与键同序列长度）

    # 2. 初始化多头注意力
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 3. 前向传播
    output, attn_weights = mha(q, k, v)

    # 4. 输出维度检查
    print(f"输入Q形状: {q.shape}")  # (2, 5, 128)
    print(f"MHA输出形状: {output.shape}")  # (2, 5, 128) （与输入Q序列长度和维度一致）
    print(
        f"注意力权重形状: {attn_weights.shape}"
    )  # (2, 8, 5, 5) （批次×头数×查询序列长×键序列长）

    # 5. 可视化第一个样本的注意力权重（展示不同头的关注模式）
    import matplotlib.pyplot as plt
    import seaborn as sns

    sample_idx = 0  # 取第一个样本
    plt.figure(figsize=(16, 10))
    for head in range(num_heads):
        plt.subplot(2, 4, head + 1)  # 8个头，2行4列布局
        # 取出第head个头的注意力权重（5×5）
        weights = attn_weights[sample_idx, head].detach().numpy()
        sns.heatmap(
            weights,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=[f"K{i}" for i in range(seq_len)],
            yticklabels=[f"Q{i}" for i in range(seq_len)],
        )
        plt.title(f"Head {head+1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行测试
    test_multi_head_attention()
