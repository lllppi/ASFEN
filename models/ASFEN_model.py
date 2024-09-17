import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.modeling_utils import BertCrossEncoder_AttnMap, BertSelfAttention

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ASFENClassifier(nn.Module):
    def __init__(self, bert, opt, config):
        super().__init__()
        self.opt = opt
        self.vision = VFIA(bert, opt=opt)
        self.gcn = GCNAbsa(bert, opt=opt)
        self.self_attention = BertSelfAttention(config)
        self.cross_attention = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.dropout = nn.Dropout(0.3)
        self.gather = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(opt.max_length, opt.polarities_dim)

    def forward(self, inputs, input_mask):
        outputs1 = self.gcn(inputs)
        outputs2 = self.vision(inputs)
        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
        self_outputs1 = self.self_attention(outputs1, extended_sent_mask)
        self_outputs2 = self.self_attention(outputs2, extended_sent_mask)
        all_encoder_layers, all_attn_maps = self.cross_attention(self_outputs1, self_outputs2, extended_sent_mask, output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers[-1]
        gathered_all_encoder_layers = self.gather(self.dropout(all_encoder_layers)).squeeze(2)
        logits = self.classifier(gathered_all_encoder_layers)
        return logits, None

class VFIA(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.gate_caption = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.gate_face = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.gate = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.drop = nn.Dropout(p=opt.vision_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, opt.num_classes)

    def forward(self, inputs):
        input_ids = inputs[8]
        attention_mask = inputs[9]
        s2_input_ids = inputs[10]
        s2_attention_mask = inputs[11]
        sequence_output, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        s2_sequence_output, s2_pooled_output = self.bert(input_ids=s2_input_ids, attention_mask=s2_attention_mask)
        caption = self.gate_caption(sequence_output)
        face_anps = self.gate_face(s2_sequence_output)
        gate_all = self.gate(caption * face_anps)
        T = torch.sigmoid(gate_all)
        C = 1 - T
        outputs_save = T * caption + C * face_anps
        outputs = self.drop(outputs_save)
        return outputs


class GCNAbsa(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)
        self.bert_dim = opt.bert_dim

    def forward(self, inputs):
        aspect_mask = inputs[6]
        h = self.gcn(inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)
        outputs1 = (h*aspect_mask).sum(dim=1) / asp_wn
        outputs = outputs1.unsqueeze(-1).repeat(1, 1, self.bert_dim)
        return outputs


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = opt.max_length
        self.W = nn.Linear(self.attdim,self.attdim)
        self.Wx = nn.Linear(self.attention_heads+self.attdim*2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim,50)
        self.aggregate_W = nn.Linear(self.attdim*2, self.attdim)

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        attention_mask = inputs[2]
        src_mask = inputs[5]
        aspect_mask = inputs[6]
        short_mask = inputs[7]
        src_mask = src_mask.unsqueeze(-2)
        batch = src_mask.size(0)
        len = src_mask.size()[2]
        text_bert_indices = text_bert_indices.cuda()
        attention_mask = attention_mask.cuda()
        sequence_output, pooled_output = self.bert(input_ids=text_bert_indices, attention_mask=attention_mask)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)  
        pooled_output = self.pooled_drop(pooled_output)
        gcn_inputs = self.Wxx(gcn_inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)
        aspect = (gcn_inputs*aspect_mask).sum(dim=1) / asp_wn
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, short_mask, aspect, src_mask)   
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None
        weight_adj = attn_tensor
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        for i in range(self.layers):
            gcn_outputs = gcn_outputs.unsqueeze(1).expand(batch, self.attention_heads, len, self.attdim)   
            Ax = torch.matmul(weight_adj, gcn_outputs)     
            Ax = Ax.mean(dim=1)
            Ax = self.W(Ax)   
            weights_gcn_outputs = F.relu(Ax)
            gcn_outputs = weights_gcn_outputs     
            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs
            weight_adj=weight_adj.permute(0, 2, 3, 1).contiguous()
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(batch, len, len, self.attdim)   
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
            node = torch.cat([node_outputs1, node_outputs2], dim=-1)
            edge_n = torch.cat([weight_adj, node], dim=-1)
            edge = self.Wx(edge_n) 
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge 
            weight_adj = edge.permute(0, 3, 1, 2).contiguous()
        outputs = torch.cat(layer_list, dim=-1)
        # node_outputs=self.Wi(gcn_outputs)
        node_outputs = self.aggregate_W(outputs)
        node_outputs = F.relu(gcn_outputs)
        return node_outputs


def attention(query, key, short, aspect, weight_m, bias_m, mask=None, dropout=None):   
    d_k = query.size(-1)   
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    batch = len(scores)
    p = weight_m.size(0)
    max = weight_m.size(1)
    weight_m = weight_m.unsqueeze(0).repeat(batch, 1, 1, 1)
    aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias_m))
    scores = torch.add(scores, aspect_scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores=torch.add(scores, short)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  
        self.d_k = d_model // h  
        self.h = h    
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k)) 
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)
    

    def forward(self, query, key, short, aspect, mask=None):   
        mask = mask[:, :, :query.size(1)]  
        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        
        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect) 
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key, short, aspect, self.weight_m, self.bias_m, mask=mask, dropout=self.dropout)
        return attn
