from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from loguru import logger


class Fusion(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        dropout: int = 0.15,
        add_norm: bool = False,
    ) -> None:
        super().__init__()
        if add_norm and (in_features != out_features):
            raise ValueError

        if hidden_features is None:
            hidden_features = 4 * in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.add_norm = add_norm
        if add_norm:
            self.layernorm = nn.LayerNorm(out_features)

    def _fusion(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

    def forward(self, x):
        if self.add_norm:
            return self.layernorm(x + self._fusion(x))
        return self._fusion(x)

class Reducer(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被执行
        super().__init__()
        # 定义一个 Transformer 编码器层
        # d_model 表示输入序列的特征维度，这里设置为 128
        # nhead 表示多头注意力机制中的头数，这里设置为 8
        # dim_feedforward 表示前馈神经网络的隐藏层维度，这里设置为 512
        # dropout 表示丢弃率，用于防止过拟合，这里设置为 0.15
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.15,
        )
        # 定义一个 Transformer 编码器，由多个编码器层堆叠而成
        # encoder_layer 表示使用的编码器层，这里使用上面定义的 self.transformer_encoder_layer
        # num_layers 表示编码器层的数量，这里设置为 2
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, num_layers=2
        )

    def forward(self, x, attention_mask):
        # 通过 Transformer 编码器处理输入序列，并使用注意力掩码来忽略填充位置
        # logger.info(f"x:{x}")
        token_embeddings = self.transformer_encoder(
            x, src_key_padding_mask=attention_mask
        )
        # logger.info(f"token_embeddings:{token_embeddings}")
        # 反转注意力掩码，将填充位置标记为 1，非填充位置标记为 0
        attention_mask_rev = (~attention_mask).int()
        # attention_mask_rev = (attention_mask).int()
        # 转置注意力掩码，使其形状与 token_embeddings 兼容
        attention_mask_rev = attention_mask_rev.transpose(1, 0)

        # 扩展注意力掩码，使其形状与 token_embeddings 相同
        input_mask_expanded = (
            attention_mask_rev.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        # 对 token_embeddings 应用注意力掩码，并在序列长度维度上求和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
        # 计算每个样本的有效位置数量
        sum_mask = input_mask_expanded.sum(0)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # 计算池化后的表示，即每个样本的有效位置的平均值
        pooled = sum_embeddings / sum_mask
        return pooled
 
class FileHunkEncoder(nn.Module):
    def __init__(self):
        super(FileHunkEncoder, self).__init__()
        self.hunk_encoder = AutoModel.from_pretrained("codebert-base")

        self.hunk_compare_linear1 = nn.Linear(768 * 2, 768)
        self.hunk_compare_linear2 = nn.Linear(768, 768)
        self.hunk_compare_concat = Fusion(768 * 3, 128, 1024)
        self.hunk_reducer = Reducer()
        self.file_reducer = Reducer()
        self.layernorm = nn.LayerNorm(768 * 3)
        
        self.classifier = nn.Linear(128, 2)

    def encode_hunk(self, input_ids, attention_mask):
        return self.hunk_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code
    
    def forward_compare_linear(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.hunk_compare_linear1(concat)
        output = F.relu(output)
        output = self.hunk_compare_linear2(output)
        return output
    
    def forward_hunk(
        self,
        hunk_add_input_ids,
        hunk_add_attention_mask,
        hunk_delete_input_ids,
        hunk_delete_attention_mask,
    ):
        n_batch, n_hunk = hunk_add_input_ids.shape[0], hunk_add_input_ids.shape[1]
        hunk_features = []

        for i in range(n_hunk):
            hunk_add_input_ids_i, hunk_add_attention_mask_i = [], [] 
            hunk_del_input_ids_i, hunk_del_attention_mask_i = [], []
            for b in range(n_batch):
                hunk_add_input_ids_i.append(hunk_add_input_ids[b][i])
                hunk_add_attention_mask_i.append(hunk_add_attention_mask[b][i])
                hunk_del_input_ids_i.append(hunk_delete_input_ids[b][i])
                hunk_del_attention_mask_i.append(hunk_delete_attention_mask[b][i])

            # 转换成张量(n_hunk, n_batch)
            hunk_add_input_ids_i = torch.stack(hunk_add_input_ids_i, dim=0)
            hunk_add_attention_mask_i = torch.stack(hunk_add_attention_mask_i, dim=0)
            hunk_del_input_ids_i = torch.stack(hunk_del_input_ids_i, dim=0)
            hunk_del_attention_mask_i = torch.stack(hunk_del_attention_mask_i, dim=0)
            # logger.info(f"1hunk_input_ids_i:shape: {hunk_add_input_ids_i.shape} , {hunk_add_input_ids_i}")
            # logger.info(f"2hunk_add_attention_mask_i:shape: {hunk_add_attention_mask_i.shape} , {hunk_add_attention_mask_i}")  
            # add和del分别encode
            hunk_add_feature_i = self.encode_hunk(
                hunk_add_input_ids_i, hunk_add_attention_mask_i
            )
            hunk_del_feature_i = self.encode_hunk(
                hunk_del_input_ids_i, hunk_del_attention_mask_i
            )

            # logger.info(f"3hunk_add_feature_i:{hunk_add_feature_i}")
            # logger.info(f"4hunk_del_feature_i:{hunk_del_feature_i}")  
            assert hunk_add_feature_i.shape == hunk_del_feature_i.shape
            assert hunk_add_feature_i.shape == torch.Size([n_batch, 768])

            hunk_feature = self.compare_hunk_features(
                hunk_add_feature_i, hunk_del_feature_i
            )  # (batch, 128)
            hunk_features.append(hunk_feature)

            # logger.info(f"3.hunk_feature_i:{hunk_del_feature_i}")  
        hunk_features = torch.stack(hunk_features, dim=1)  # (batch, hunk, 128)
        return hunk_features
    
    def compare_hunk_features(self, hunk_add_feature, hunk_del_feature):
            sub = self.subtraction(hunk_add_feature, hunk_del_feature)  # (batch, 768)
            # mul = self.multiplication(hunk_add_feature, hunk_del_feature)  # (batch, 768)
            # lin = self.forward_compare_linear(
            #     hunk_add_feature, hunk_del_feature
            # )  # (batch, 768)
            out = self.layernorm(torch.cat((sub, hunk_add_feature, hunk_del_feature), dim=1))


            out = self.hunk_compare_concat(out)
            return out

    def forward(
        self,
        code_add_input_ids,
        code_add_attention_mask,
        code_del_input_ids,
        code_del_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
    ):
        n_batch, n_file, n_hunk = (
            code_add_input_ids.shape[0],
            code_add_input_ids.shape[1],
            code_add_input_ids.shape[2],
        )

        code_features = []
        for i in range(n_file): # (n_file,n_batch)
            file_add_input_ids_i, file_add_attention_mask_i = [], []
            file_del_input_ids_i, file_del_attention_mask_i = [], []
            for b in range(n_batch):
                file_add_input_ids_i.append(code_add_input_ids[b][i])
                file_add_attention_mask_i.append(code_add_attention_mask[b][i])
                file_del_input_ids_i.append(code_del_input_ids[b][i])
                file_del_attention_mask_i.append(code_del_attention_mask[b][i])

            file_add_input_ids_i = torch.stack(file_add_input_ids_i, dim=0)
            file_add_attention_mask_i = torch.stack(file_add_attention_mask_i, dim=0)
            file_del_input_ids_i = torch.stack(file_del_input_ids_i, dim=0)
            file_del_attention_mask_i = torch.stack(file_del_attention_mask_i, dim=0)

            code_feature = self.forward_hunk(
                file_add_input_ids_i,
                file_add_attention_mask_i,
                file_del_input_ids_i,
                file_del_attention_mask_i,
            )
            code_features.append(code_feature)
        # 一个提交中所有file的hunk特征
        code_features = torch.stack(code_features, dim=1)
        # logger.info(f"code_features:{code_features}")
        assert code_features.shape == torch.Size([n_batch, n_file, n_hunk, 128])  

        files = None
        for f in range(n_file):
            hunks = []
            hunk_attention_masks = []
            for b in range(n_batch):
                hunk_attention_masks.append(hunk_attention_mask[b][f])
            for h in range(n_hunk):
                hunk = []
                for b in range(n_batch):
                    hunk.append(code_features[b][f][h])
                hunk = torch.stack(hunk, dim=0)
                assert hunk.shape == torch.Size([n_batch, 128])
                hunks.append(hunk)
            hunks = torch.stack(hunks, dim=0)
            hunk_attention_masks = torch.stack(hunk_attention_masks, dim=0)
            assert hunks.shape == torch.Size([n_hunk, n_batch, 128])
            assert hunk_attention_masks.shape == torch.Size([n_batch, n_hunk])
            hunks_feature = self.hunk_reducer(hunks, hunk_attention_masks)
            hunks_feature = hunks_feature.unsqueeze(dim=0)

            if files is None:
                files = hunks_feature
            else:
                files = torch.cat((files, hunks_feature), dim=0)
        assert files.shape == torch.Size([n_file, n_batch, 128]), files.shape

        commit_feature = self.file_reducer(files, file_attention_mask)
        assert commit_feature.shape == torch.Size([n_batch, 128])
        # logger.info(f"commit feature:{commit_feature}")
        # return commit_feature
        x = self.classifier(commit_feature)
        return commit_feature


class CommitMessageModel(nn.Module):
    def __init__(self):
        super(CommitMessageModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("codebert-base")
        self.classifier = FeedForward(768, 768, 1536)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_batch, mask_batch):
        embeddings = self.encoder(input_ids=input_batch, attention_mask=mask_batch).pooler_output # 选择cls的embedding
        x = self.classifier(embeddings)
        out = self.out_proj(x)
        return embeddings

class VFHClassifier(nn.Module):
    def __init__(self, code_change_encoder_params, msg_encoder_params) -> None:
        super().__init__()
        # self.config = config
        self.code_change_encoder = FileHunkEncoder()
        self.code_change_encoder.load_state_dict(code_change_encoder_params)
        for param in self.code_change_encoder.parameters():
            param.requires_grad = False

        self.msg_encoder = CommitMessageModel()
        self.msg_encoder.load_state_dict(msg_encoder_params)
        for param in self.msg_encoder.parameters():
            param.requires_grad = False


        self.text_code_combiner = Fusion(768 + 128, 768, 2048)

        self.classifier = nn.Linear(768, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
    ):
        msg_embeding = self.msg_encoder(
            input_ids,
            attention_mask,
        )

        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        embedding = torch.cat((msg_embeding, code_embeding), dim=1)

        combined = self.text_code_combiner(
            embedding
        )

        return self.classifier(combined)




