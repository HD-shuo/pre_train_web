import math
from typing import Optional, Tuple, Union
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    在注意力机制的过程中将多个注意力头的输出组合在一起，以便后续计算
    :para
        hidden_states:torch.Tensor,表示模型隐层状态，形状为(batch, num_key_value_heads, slen, head_dim)
            batch:批次大小
            num_key_value_heads:注意力多头数量
            seqlen:序列长度
            head_dim:每个注意力头的维度
        n_rep:int,表示要在指定维度重复多少次
    :return
        hidden_states:综合多头注意力之后的输出，reshape后的张量大小为(batch, num_key_value_heads * n_rep, slen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 检查n_rep是否等于1, 若不等于,则扩展hidden_states张量维度，将维度n_rep插入到原来的hidden_states的第二个维度之后
    # reshape hidden_states,将张量的形状变为(batch, num_key_value_heads * n_rep, slen, head_dim)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    将张量x的前一半维度和后一半维度进行交换
    """
    
    # 此处逻辑与原始的ROPE有所差异，原始逻辑如下
    # 这里本质上是从最后一个维度开始取了原始张量奇数位置和偶数位置的值，把二者在最后一个维度交替拼接，实现维度旋转
    #x1 = x[..., 0::2] # 这里的截取操作步长为2，从0开始，本质上是截取了x所有偶数位置的元素
    #x2 = x[..., 1::2] # 这里是取奇数位置元素
    #res = torch.cat((x1, x2), dim=-1)
    #res[...,0::2]=-x2 # 将奇数位置元素取相反数插入到res的偶数位置
    #res[...,1::2]=x1 # 将偶数位置元素取出插入到res的奇数位置
    #return res
    
    #从x的最后一个维度开始截取前一半的数据
    x1 = x[..., : x.shape[-1] // 2] 
    #从x的最后一个维度开始截取后一半的数据
    x2 = x[..., x.shape[-1] // 2 :]
    # 在最后一个维度上拼接-x2与x1,实现维度旋转
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    将rope位置编码嵌入到q,k矩阵中去
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.换言之可以把第一维和第二维去掉
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # 根据position_id从cos与sin中选取相应位置的值，这是通过索引操作实现的
    # 该位置的值是一个张量，大小为[bs, 1, seq_len, dim]，因为有bs维度，换言之每个样本的位置编码被广播
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEEmbedding(nn.Module):
    """
    Implement positional encoding of inputs
    :para
        dim:嵌入的维度大小
        max_position_embeddings:序列的最大长度(文本中的最大词数或标记数), 默认为2048
        base:计算相对位置编码的基数, 默认为10000
        device:表示在哪个设备上创建模块, 默认是None
    :return
        cos_cached:计算得到的余弦编码
        sin_cached:计算得到的正弦编码
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # inv_freq是一个张量, 值是0.1除以一个序列中每个位置的频率，这个频率是根据base参数计算得到的，这个张量用于计算RoPE的正弦和余弦部分
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 用于将张量inv_freq注册为模块的缓冲区，意味着它不会出现在模型的可训练参数中(即不会在模型的state_dict中保存)，而是随着模块一起移动到GPU或CPU上
        self.register_buffer("inv_freq", inv_freq, persistent=False) #persistent=False将不会作为state_dict

        # Build here to make `torch.jit.trace` work.
        # 设置RoPE的正弦和余弦缓存，根据相对位置编码的公式进行计算，在后续的前向传播中会被继续使用
        self._set_cos_sin_cache(
        seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 创建了一个长度为seq_len的序列t, 代表了相对位置索引
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # 将t乘以inv_freq，计算频率信息,这里使用了torch.einsum()函数，"i,j->ij"是一个爱因斯坦求和约定，表示对t和inv_freq进行相乘
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 编码信息emb由正弦和余弦部分组成
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将计算得到的正弦和余弦编码缓存为模块的缓冲区
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 超过预设的max_position_embeddings则重新计算更大的Rope缓存，否则直接在缓存上切片
        # 计算RoPE的正弦和余弦部分，接受输入的x和可选的seq_len参数，表示要计算的序列长度
        # 如果 seq_len 大于之前缓存的最大序列长度 max_seq_len_cached，则调用 _set_cos_sin_cache 方法重新计算更大的RoPE缓存。
        # 最后，方法返回计算得到的正弦和余弦编码。
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        used rmsnorm is same as LlamaRMSNorm
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AttentionModule(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    相较llama使用的attention模式,删除了关于rope_scaling判定的部分,即不做长度扩展
    k和v的head数量可以是q的head数量的几分之一，类似分组卷积的思想，可以减少参数规模。
    rope位置编码是每次做多头注意力时都进行一次，而不是原论文只在输入的时候进行一次。
    允许传入key和value的states的缓存past_key_value，这在多轮对话中可以减少重复计算，起到加速效果。
    attention_mask是通过加法形式作用到softmax之前的attention矩阵上的。
    :params
        config params
            hidden_size:隐层大小
            num_attention_heads:注意力头数量
            num_key_value_heads:键值对注意力头数量
            max_position_embeddings:最大位置嵌入
        head_dim:每个注意力头的维度,一般是hidden_size // num_heads
        num_key_value_groups:计算用于键值对的注意力头的组数,一般是num_heads // num_key_value_heads
    :return
        attn_output:注意力输出
        attn_weights:注意力矩阵
        past_key_value:历史kv
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # 检查隐层是否均匀地分给了每个注意力头，否则报错
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 创建q,k,v矩阵。将隐层状态通过线性层分别投射到q,k,v空间
        # q空间维度：num_heads * head_dim
        # k空间维度：num_key_value_heads * head_dim
        # v空间维度：num_key_value_heads * head_dim
        # 再创建一个线性层将多头注意力输出投影回原始隐层空间，输出尺寸为hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
    
    # 初始化位置嵌入(私有方法)
    def _init_rope(self):
        self.rotary_emb = RoPEEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    # 重新整形tensor大小 将输入张量整形为(bsz, seq_len, self.num_heads, self.head_dim)
    # 再将第一与第二个维度交换，即(bsz, self.num_heads, seq_len, self.head_dim)
    # contiguous()用于确保张量连续
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 获取输入张量hidden_state的形状信息
        bsz, q_len, _ = hidden_states.size()

        # 检查配置参数pretraining_tp，如果大于1就进行预处理
        # 预处理的本质是将hidden_state分成了多个qkv头，每个头有自己独立的向量，这样就能并行地计算每个头的注意力分数，提高运行效率
        if self.config.pretraining_tp > 1:
            # 计算k-v切片大小，本质是将k,v矩阵维度除以pretraining_tp获得切片大小
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            # 查询/键/值头的张量将分成多个部分，分别进行线性变换。然后，这些部分将被连接起来，以形成最终的查询/键/值张量。
            # 获得q矩阵切片，本质是将q的投影矩阵在第0维(行)切分为(self.num_heads * self.head_dim) // self.config.pretraining_tp大小
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            # 获得k、v切片大小
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            # 遍历pretraining_tp个切片，将输入的hidden_state与qkv相应大小的切片去进行线性变换，相当于为每个头计算其对应的向量
            # 再通过torch.cat将所有的投影结果拼接
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            # 若不进行预处理，将输入的hidden_states通过线性变换分别得到查询、键和值的张量。
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 通过view和transpose，将q、k和v的张量重新整形，以满足注意力机制的要求
        # 整形后的形状会变成 `(bsz, num_heads, q_len, head_dim)`，这是为了将注意力头数放在中间的维度
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 获取kv的序列长度，用于RoPE的初始化(对于k这个四维张量,第二维是q_len,及序列长度)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # 使用 RoPEEmbedding计算余弦和正弦编码。这些编码将被用于在注意力计算中应用旋转。
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # 应用旋转位置编码到查询和键的张量上，以改善注意力计算
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 这里是考虑到了过去的信息
        # 如果存在过去的键/值张量 (`past_key_value`)，则它们将被重新连接到当前的键/值张量中
        # 具体的方式是将包含过去信息的k、v矩阵在dim=2维度上与当前k、v张量拼接
        # 这是为了允许在生成序列时重用之前的信息，以提高效率
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 如果use_cache为True,则把k,v组成的元组赋值给past_key_value
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        # 将键和值的头部张量进行重复以匹配总头部数。这是为了处理当 `num_key_value_heads` 小于 `num_heads` 时的情况
        # 根据repeat_kv函数的定义，实际是把num_key_value_groups组输出的k,v矩阵进行整合得到新的k,v矩阵
        # 我的理解是因为kv头的数量只有q的几分之一，num_key_value_groups实际上是q头是kv头的倍数，所谓的repeat的含义也就是整合num_key_value_groups组kv头使之维度与q头一致，然后一起输出

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算q-k注意力权重，衡量q-k之间的关联程度
        # query_states:(bsz, num_heads, q_len, head_dim)
        # key_states:(bsz, num_key_value_heads, q_len, head_dim)
        # 此处将k矩阵进行转置使之能与q矩阵进行矩阵相乘
        # 注意力矩阵缩放因子：math.sqrt(self.head_dim),取了head_dim的平方根倒数作为缩放因子
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # q-k权重矩阵大小检查
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 如果存在attention_mask，检查其尺寸无误后将其添加到attn_weights中，以屏蔽某些位置的注意力
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # 通过softmax操作将注意力权重标准化，得到最终的注意力输出attn_output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 将注意力输出尺寸转置为(bsz, q_len, num_heads, kv_seq_len)，再reshape回隐层大小
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 根据pretraining_tp选择对注意力输出的进一步操作
        # 如果大于1，就在隐层尺寸维度上取出attn_output的切片以及o矩阵的权重矩阵切片，再分别对每个切片应用线性变换（o_proj）得到最终的输出
        # 如果小于1，就正常将注意力输出通过o矩阵映射回原本隐层空间
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        # 如果不需要输出注意力权重（output_attentions=False），则attn_weights被设为None
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class MLP(nn.Module):
    """
    定义了一个两层的感知机MLP,先从hidden_size维度up_proj到intermediate_size维度,然后再down_proj还原为hidden_size维度
    此外定义了一个门控映射函数
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 如果pretraining_tp > 1,与之前计算注意力类似，将intermediate_size切片成pretraining_tp份
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            # 分别在维度0上对门控投影和上采样投影矩阵进行切片，再在维度1上对下采样投影矩阵切片
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # 将投影后的门控切片序列在最后一个维度上进行拼接
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            # 将投影后的上采样切片序列在最后一个维度上进行拼接
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            # 中间层状态通过对门控序列进行激活后与上采样投影相乘得到，然后再在第二个维度上进行切片
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            # 下采样投影通过遍历中间层状态的每个切片并映射到下采样矩阵切片上得到，最后的down_proj是所有下采样切片的和
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # 如果pretraining_tp < 1,则将输入直接过门控层激活后进行上采样然后再下采样
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    

class EncoderLayer(nn.Module):
    """
        构造编码层:self_attention + feed_forward
        此处encoder_layer = norm + attention + norm + mlp
    :params(config)
        hidden_size:隐层大小
        rms_norm_eps:用于数值稳定的超参数,eps被添加到分母,用于防止分母接近于0的情况
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = AttentionModule(config=config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self, 
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # 如果需要注意力权重以及use_cache,就把这两个加到输出的元组里
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Encoder(PreTrainedModel):
    """
    设置gradient_checkpoint=True来节省显存
    其主要应用了torch.utils.checkpoint.checkpoint方法
    在对encoder_layer进行foward时不保存中间激活值来节约显存
    在backward时再重新计算，是一种时间换空间的策略
    """
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpoint = False

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions

        hidden_states = inputs_embeds

        # decoder layers
        # 如果output_hidden_states为真，则初始化一个空的output_hidden_states,否则将其初始化为None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None


        for encode_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            if self.gradient_checkpoint and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encode_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = encode_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache = use_cache,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last encoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states, )
            
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states,all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    

class WebPreTrainedModel(PreTrainedModel):
    """
    定义预训练模型基类，继承自huggingface Transformer库
    """
    # 模型状态字典中与模型相关的键的前缀，通常是一个字符串，用于检查点在文件中标识模型的参数
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 一个列表，包含不希望进行梯度分割的模块的名称
    _no_split_modules = ["EncoderLayer"]

    # 初始化模型权重，将其初始化为指定的均值与标准差
    def _init_weights(self, module):
        std = self.config.initializer_range
        # 线性层权重初始化为服从正态分布的随机值，均值为0，标准差为std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 是否启用梯度检查点，主要用于将指定模块的gradient_checkpointing属性指定为特定的值
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.gradient_checkpointing = value


class PredictionHeadTransform(nn.Module):
    """
    转换预测头的输出
    包含一个全连接层+激活层+归一化层
    """
    def __init__(self, config) -> None:
        super(PredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 检查config.hidden_act是否是字符串类型或者(python版本为2.X)是Unicode类型
        if isinstance(config.hidden_act, str) or (sys.version_info[0]==2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        self.LayerNorm = RMSNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

class LMPredictionHead(nn.Module):
    """
    构造了一个LM预测头
    包含一个预测transform层+decoder
    这个decoder本质也是一个线性层，输入维度是model_embedding_weights.size(1)，输出维度是model_embedding_weights.size(0)
    作用是将模型的隐藏状态映射到词汇表大小的向量，以便能够进行下游的语言任务
    """
    def __init__(self, config, model_embedding_weights):
        super(LMPredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config=config)
        self.decoder = nn.Linear(model_embedding_weights.size(1),
                                 model_embedding_weights.size(0),
                                 bias=False)
        
        # 这里把模型的嵌入权重初始化为decoder的权重(迁移学习)
        self.decoder.weight = model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(model_embedding_weights.size(0)))
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    

class LMPreTrainingHeads(nn.Module):
    """
    构造LM预训练预测头
    基于上文的LMPredictionHead构造了predictions方法用于下游语言建模预测
    另外构造了一个线性层将隐层尺寸从hidden_size映射到2，此处用于句间关系预测
    在前向传播层中有两个输入，一个是sequence_output，表示模型的序列输出
    另一个是pooled_output,模型的池化输出，用于句子级别的预测表示
    """
    def __init__(self, config, model_embedding_weights) -> None:
        super(LMPreTrainingHeads, self).__init__()
        self.predictions = LMPredictionHead(config, model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    

class WebEncoderForPreTraining(WebPreTrainedModel):
    def __init__(self, config):
        super(WebEncoderForPreTraining, self).__init__(config)
        self.encoder = Encoder(config)
        self.cls = LMPreTrainingHeads(config, self.encoder.embed_tokens.weight)
        self.apply(self._init_weights)
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            masked_lm_labels: Optional[bool] = None,
            ):
        output = self.encoder(input_ids, position_ids, inputs_embeds,use_cache,output_attentions,output_hidden_states,attention_mask,return_dict,)
        sequence_output = output[0]
        prediction_scores = self.cls(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            pass

    