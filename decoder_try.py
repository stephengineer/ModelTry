class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        # 初始化时调用父类的构造函数，确保正确继承
        super().__init__()
 
        # 从配置 `config` 中获取隐藏层大小，用于层的初始化
        self.hidden_size = config.hidden_size

        # 自注意力机制层的初始化，使用 LlamaAttention。它会处理输入的自注意力操作。
        self.self_attn = LlamaAttention(config=config)
 
        # 多层感知机层（MLP）用于后续的非线性变换。其包含一个隐藏层大小和中间层大小。
        # `hidden_act` 是激活函数（如 ReLU），用于非线性变换。
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,  # 输入的隐藏层大小
            intermediate_size=config.intermediate_size,  # 中间层大小，通常是隐藏层大小的 4 倍
            hidden_act=config.hidden_act  # 激活函数，如 ReLU 或 GELU
        )
 
        # 输入层的归一化层，使用 RMSNorm，防止训练中的梯度爆炸或消失
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
        # 注意力后的归一化层，确保经过注意力机制后的输出有稳定的分布
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量，形状为 [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，防止模型关注到不需要的序列位置
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码的ID，用于生成位置信息
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 用于缓存前面层的键值对，加速推理
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重
        use_cache: Optional[bool] = False,  # 是否使用缓存，加速推理
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 保存输入的隐藏状态到 `residual`，用于残差连接
        residual = hidden_states
 
        # 对输入的隐藏状态应用归一化处理，确保输入的数值稳定
        hidden_states = self.input_layernorm(hidden_states)
 
        # 自注意力机制，处理输入序列中的各个位置之间的相互依赖关系
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,  # 当前的隐藏状态
            attention_mask=attention_mask,  # 注意力掩码，防止模型关注到不相关的部分
            position_ids=position_ids,  # 位置编码，帮助模型知道序列中每个词的位置
            past_key_value=past_key_value,  # 用于加速推理的缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            use_cache=use_cache,  # 是否缓存键值对，加速下一步计算
        )
 
        # 通过残差连接，将输入（residual）与注意力层的输出相加，以避免梯度消失问题
        hidden_states = residual + hidden_states
 
        # 经过注意力机制后，保存当前的隐藏状态作为残差
        residual = hidden_states
 
        # 对经过注意力后的隐藏状态再次进行归一化处理
        hidden_states = self.post_attention_layernorm(hidden_states)
 
        # 进入多层感知机（MLP）模块，进行非线性变换，增加网络的表达能力
        hidden_states = self.mlp(hidden_states)
 
        # 再次通过残差连接，将 MLP 的输出与注意力后的隐藏状态相加
        hidden_states = residual + hidden_states
 
        # 将隐藏状态作为输出
        outputs = (hidden_states,)
 
        # 如果需要输出注意力权重，则将注意力权重也加入输出
        if output_attentions:
            outputs += (self_attn_weights,)
 
        # 如果使用缓存，则将当前层的键值对缓存下来，便于后续层使用
        if use_cache:
            outputs += (present_key_value,)
 
        # 返回包含隐藏状态和其他可选输出的元组
        return outputs