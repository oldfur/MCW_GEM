import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionMLP(nn.Module):
    def __init__(self, 
                 input_dim=4,      # 输入数据维度
                 output_dim=4,     # 输出噪声维度（通常与输入相同）
                 time_emb_dim=32,  # 时间步嵌入维度
                 hidden_dims=[256, 256, 512],  # 隐藏层结构
                 dropout=0.1,      # Dropout率
                 use_self_attn=True):  # 是否使用注意力
        super().__init__()
        
        # 时间步编码器（正弦位置编码）
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 输入投影层（将输入+时间编码融合）
        self.input_proj = nn.Linear(input_dim + time_emb_dim, hidden_dims[0])
        
        # 构建隐藏层（带跳跃连接）
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.blocks.append(ResBlock(
                dim_in=hidden_dims[i],
                dim_out=hidden_dims[i+1],
                time_emb_dim=time_emb_dim,
                dropout=dropout
            ))
        
        # 可选的自注意力层（用于全局依赖）
        if use_self_attn:
            self.attn = SelfAttentionBlock(hidden_dims[-1])
        else:
            self.attn = None
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 正交初始化更适合扩散模型
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        """
        输入:
        - x: 噪声数据 [batch, input_dim]
        - t: 时间步 [batch, 1] (值范围[0,1])
        """
        # 时间编码 [batch, time_emb_dim]
        dtype = self.time_embed[0].weight.dtype
        t_emb = self.time_embed(t.to(dtype))
        
        # 融合输入和时间编码
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)
        
        # 通过残差块
        for block in self.blocks:
            h = block(h, t_emb)
        
        # 自注意力
        if self.attn is not None:
            h = h.unsqueeze(1)  # [batch, 1, dim]
            h = self.attn(h)
            h = h.squeeze(1)
        
        # 输出噪声预测
        return self.output_layer(h)

class ResBlock(nn.Module):
    """带时间步嵌入的残差块"""
    def __init__(self, dim_in, dim_out, time_emb_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        )
        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out, dim_out),
            nn.LayerNorm(dim_out)
        )
        self.res_conv = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.block(x) + self.mlp(t_emb)
        return h + self.res_conv(x)

class SelfAttentionBlock(nn.Module):
    """轻量级自注意力"""
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, -1), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(B, L, D)
        
        return self.norm(self.proj(out) + x)

if __name__ == "__main__":
    # 初始化网络 (输入3维，输出3维噪声)
    model = DiffusionMLP(input_dim=3, output_dim=3, 
                         hidden_dims=[128, 128],  # 较浅网络
                         use_self_attn=False # 关闭注意力
                        )

    # 模拟输入 (batch_size=16)
    x_noisy = torch.randn(16, 3)  # 带噪声数据
    timesteps = torch.rand(16, 1)  # 随机时间步 [0,1]

    # 预测噪声
    predicted_noise = model(x_noisy, timesteps)
    print(predicted_noise.shape)  # [16, 3]