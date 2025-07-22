import torch.nn as nn
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights
from transformers import ViTModel


class Multi_Label_ViT_Net(nn.Module):
    def __init__(self, num_classes=8, hidden_dim=1024, fusion_num_heads=16):
        '''
        初始化 Multi_Label_ViT_Net

        Args：
            :param num_classes: 类别数
            :param hidden_dim: 分类器隐藏层维度
            :param fusion_num_heads: fusion attention 的头数
        '''
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_num_heads = fusion_num_heads

        # 初始化 encoder
        # self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)     # ViT_Base_16
        self.encoder = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)        # ViT_Large_16
        # self.encoder = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
        # self.encoder = ViTModel.from_pretrained("pretrain_models/ViT_Huge_224")
        # embed_dim = self.encoder.config.hidden_size
        embed_dim = self.encoder.hidden_dim

        # Attention fusion
        # self.fusion_attention = nn.MultiheadAttention(embed_dim=self.encoder.hidden_dim, num_heads=fusion_num_heads)
        self.fusion_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=fusion_num_heads)

        # 多分类头
        self.classifier_N = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_D = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_G = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_C = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_A = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_H = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_M = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier_O = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x_l, x_r):
        '''
        forward

        Args:
            :param x_l: 左图像
            :param x_r: 右图像

        :return:
        '''
        self.encoder.heads = nn.Identity()      # 去除 encoder 分类头
        # encoder
        latent_l_cls = self.encoder(x_l)
        latent_r_cls = self.encoder(x_r)
        # latent_l_cls = self.encoder(x_l).last_hidden_state[:, 0, :]
        # latent_r_cls = self.encoder(x_r).last_hidden_state[:, 0, :]

        # fusion(attention)
        # 将class token堆叠为序列 (2, B, D)
        combined = torch.stack([latent_l_cls, latent_r_cls], dim=0)
        # 注意力计算 (query, key, value都用同一个)
        attn_output, _ = self.fusion_attention(combined, combined, combined)
        # 拼接注意力输出 (B, 2*D)
        latent_cls = torch.cat([attn_output[0], attn_output[1]], dim=1)

        # classify
        logits_N = self.classifier_N(latent_cls)
        logits_D = self.classifier_D(latent_cls)
        logits_G = self.classifier_G(latent_cls)
        logits_C = self.classifier_C(latent_cls)
        logits_A = self.classifier_A(latent_cls)
        logits_H = self.classifier_H(latent_cls)
        logits_M = self.classifier_M(latent_cls)
        logits_O = self.classifier_O(latent_cls)
        logits = {
            "N": logits_N,
            "D": logits_D,
            "G": logits_G,
            "C": logits_C,
            "A": logits_A,
            "H": logits_H,
            "M": logits_M,
            "O": logits_O
        }

        return logits

