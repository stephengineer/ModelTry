import torch
import torch.nn as nn
import math


class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (Grouped Query Attention)
    
    GQA是一种高效的注意力机制，多个查询头共享相同的键和值头，
    从而减少内存使用和计算量，同时保持接近MHA的性能。
    """

    def __init__(self, d_model, num_heads, num_kv_heads=None, d_k=None):
        super().__init__()
        self.d_model = d_model  # 模型总维度
        self.num_heads = num_heads  # 查询头数
        self.num_kv_heads = num_kv_heads or num_heads  # 键值头数，默认为查询头数
        self.d_k = d_k or d_model // num_heads  # 每个头的维度
        
        # 计算分组大小：每个键值头对应多少个查询头
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        assert num_heads % self.num_kv_heads == 0, "查询头数必须能被键值头数整除"
        
        # 定义投影层
        self.w_q = nn.Linear(d_model, num_heads * self.d_k)  # 查询投影
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)  # 键投影
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)  # 值投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力计算"""
        # q: (batch_size, num_heads, seq_len_q, d_k)
        # k: (batch_size, num_kv_heads, seq_len_k, d_k)
        # v: (batch_size, num_kv_heads, seq_len_v, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 1. 投影到查询、键、值空间
        q_proj = self.w_q(q)  # (batch_size, seq_len, num_heads * d_k)
        k_proj = self.w_k(k)  # (batch_size, seq_len, num_kv_heads * d_k)
        v_proj = self.w_v(v)  # (batch_size, seq_len, num_kv_heads * d_k)
        
        # 2. 重塑为多头格式
        q = q_proj.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k_proj.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = v_proj.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 3. 扩展键和值以匹配查询头数
        # 将每个键值头复制给对应的查询头组
        k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # 4. 计算注意力
        output, attn_weights = self.scaled_dot_product_attention(
            q, k_expanded, v_expanded, mask
        )
        
        # 5. 重塑并输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """传统多头注意力（用于对比）"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


def test_grouped_query_attention():
    """测试分组查询注意力"""
    print("=" * 60)
    print("测试分组查询注意力 (Grouped Query Attention)")
    print("=" * 60)
    
    # 配置参数
    batch_size = 2
    seq_len = 8
    d_model = 128
    num_heads = 8  # 查询头数
    num_kv_heads = 2  # 键值头数（4:1的压缩比）
    
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化GQA
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    
    # 前向传播
    output, attn_weights = gqa(q, k, v)
    
    # 输出信息
    print(f"输入形状: {q.shape}")
    print(f"GQA输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"查询头数: {num_heads}")
    print(f"键值头数: {num_kv_heads}")
    print(f"压缩比: {num_heads // num_kv_heads}:1")
    
    return gqa, output, attn_weights


def test_multi_head_attention():
    """测试传统多头注意力（用于对比）"""
    print("=" * 60)
    print("测试传统多头注意力 (Multi-Head Attention)")
    print("=" * 60)
    
    # 配置参数
    batch_size = 2
    seq_len = 8
    d_model = 128
    num_heads = 8
    
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化MHA
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 前向传播
    output, attn_weights = mha(q, k, v)
    
    # 输出信息
    print(f"输入形状: {q.shape}")
    print(f"MHA输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"头数: {num_heads}")
    
    return mha, output, attn_weights


def compare_mha_gqa():
    """比较MHA和GQA的性能和内存使用"""
    print("=" * 60)
    print("MHA vs GQA 性能对比")
    print("=" * 60)
    
    # 配置参数
    batch_size = 4
    seq_len = 16
    d_model = 256
    num_heads = 8
    num_kv_heads = 2  # 4:1压缩比
    
    # 创建相同的输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化模型
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    
    # 计算参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mha_params = count_parameters(mha)
    gqa_params = count_parameters(gqa)
    
    print(f"模型配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  模型维度: {d_model}")
    print(f"  查询头数: {num_heads}")
    print(f"  键值头数 (GQA): {num_kv_heads}")
    print()
    
    print(f"参数数量对比:")
    print(f"  MHA参数: {mha_params:,}")
    print(f"  GQA参数: {gqa_params:,}")
    print(f"  参数减少: {((mha_params - gqa_params) / mha_params * 100):.1f}%")
    print()
    
    # 计算内存使用（近似）
    # K和V的投影层参数减少
    kv_params_saved = (d_model * d_model) * 2  # K和V投影层
    print(f"键值投影层参数节省: {kv_params_saved:,}")
    print()
    
    # 测试前向传播
    with torch.no_grad():
        mha_output, mha_weights = mha(q, k, v)
        gqa_output, gqa_weights = gqa(q, k, v)
    
    print(f"输出形状对比:")
    print(f"  MHA输出: {mha_output.shape}")
    print(f"  GQA输出: {gqa_output.shape}")
    print(f"  输出形状一致: {mha_output.shape == gqa_output.shape}")
    print()
    
    print(f"注意力权重形状对比:")
    print(f"  MHA权重: {mha_weights.shape}")
    print(f"  GQA权重: {gqa_weights.shape}")
    print()


def visualize_attention_patterns():
    """可视化注意力模式"""
    print("=" * 60)
    print("注意力模式可视化")
    print("=" * 60)
    
    # 配置参数
    batch_size = 1
    seq_len = 6
    d_model = 64
    num_heads = 4
    num_kv_heads = 2
    
    # 创建输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化模型
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    
    # 前向传播
    with torch.no_grad():
        output, attn_weights = gqa(q, k, v)
    
    # 可视化
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 8))
        
        # 显示GQA的注意力权重
        for head in range(num_heads):
            plt.subplot(2, 4, head + 1)
            weights = attn_weights[0, head].detach().numpy()
            sns.heatmap(
                weights,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                xticklabels=[f"K{i}" for i in range(seq_len)],
                yticklabels=[f"Q{i}" for i in range(seq_len)],
            )
            plt.title(f"GQA Head {head + 1}")
        
        plt.tight_layout()
        plt.show()
        
        print("注意力模式可视化完成！")
        print(f"注意：在GQA中，头{1}-{2}和头{3}-{4}分别共享相同的键值头")
        
    except ImportError:
        print("需要安装matplotlib和seaborn来可视化注意力模式")
        print("pip install matplotlib seaborn")


if __name__ == "__main__":
    # 运行所有测试
    test_grouped_query_attention()
    print()
    
    test_multi_head_attention()
    print()
    
    compare_mha_gqa()
    
    visualize_attention_patterns()
