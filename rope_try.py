import torch

class LlamaRotaryEmbedding(torch.nn.Module):
    # 生成旋转矩阵
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类的初始化函数，确保模块正确继承 nn.Module 的功能
        super().__init__()
 
        # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
        # `inv_freq` 是逆频率的计算结果，它根据维度 `dim` 来生成。该逆频率用于生成正弦和余弦嵌入。
        # `torch.arange(0, dim, 2)` 生成从 0 到 dim 的偶数序列（步长为 2），然后除以 dim，构造分布式频率。
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # 将逆频率 `inv_freq` 注册为模型的缓冲区，这意味着它不是可训练参数，但会被持久保存。
        self.register_buffer("inv_freq", inv_freq)
 
        # 初始化时，预先缓存最大序列长度对应的旋转嵌入，避免在每次前向传播时重复计算
        self.max_seq_len_cached = max_position_embeddings
 
        # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
        # `t` 是时间步（位置索引），从 0 到 `max_seq_len_cached - 1` 的序列，用来生成位置嵌入。
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 通过 `einsum` 进行矩阵乘法，将 `t` 和 `inv_freq` 相乘生成频率矩阵 `freqs`，表示每个位置和对应的频率。
        # einsum("i,j->ij") 表示进行外积操作，将 `t` 和 `inv_freq` 组合成位置-频率矩阵。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
 
        # 将 `freqs` 进行拼接，扩展为 [seq_len, dim]，这样每个位置都有对应的频率嵌入。
        emb = torch.cat((freqs, freqs), dim=-1)
 
        # 获取当前默认的 `dtype`，以确保缓存的 `cos` 和 `sin` 的数据类型与输入一致。
        dtype = torch.get_default_dtype()
 
        # 缓存 cos 和 sin 嵌入，这里为嵌入增加了额外的维度以适配多头注意力机制的输入格式。
        # `persistent=False` 表示这些缓冲区不会被持久保存到模型的状态字典中。
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
 
    def forward(self, x, seq_len=None):
        # `x` 是输入张量，通常形状为 [batch_size, num_attention_heads, seq_len, head_size]。
        # 这部分代码检查当前序列长度是否超过缓存的最大序列长度 `max_seq_len_cached`。
        if seq_len > self.max_seq_len_cached:
            # 如果输入的序列长度超过了缓存的最大序列长度，重新计算 sin 和 cos 值。
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            # 重新计算位置和频率的外积。
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # 生成新的频率嵌入，并更新缓冲区。
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
 
        # 截取缓存的 cos 和 sin 值，使其匹配输入序列的长度 `seq_len`，并确保数据类型与输入一致。
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
    # 旋转位置编码计算
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)
 
# `rotate_half` 函数实现对输入张量的一半进行旋转操作，这是 RoPE 的核心机制。
# 它将输入张量的后半部分取负并与前半部分交换，形成旋转效果。
def rotate_half(x):
    """旋转输入张量的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]  # 获取输入的前半部分。
    x2 = x[..., x.shape[-1] // 2 :]  # 获取输入的后半部分。
    return torch.cat((-x2, x1), dim=-1)  # 交换并将后半部分取负，拼接成新的张量。
 
# `apply_rotary_pos_emb` 函数将旋转位置嵌入应用到查询 `q` 和键 `k` 上。
# 它将 cos 和 sin 值乘以查询和键，然后通过旋转操作进行位置嵌入的应用。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # `cos` 和 `sin` 的前两维始终为 1，因此可以去掉这些冗余维度。
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
 
    # 通过 `position_ids` 选择相应的嵌入，并扩展维度以匹配查询和键的形状。
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
 
    # 对查询 `q` 应用旋转位置嵌入，先乘以 cos，再加上乘以 sin 的旋转结果。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 对键 `k` 应用相同的旋转位置嵌入。
    k_embed = (k * cos) + (rotate_half(k) * sin)
 
    # 返回嵌入了位置编码的查询和键。
    return q_embed, k_embed