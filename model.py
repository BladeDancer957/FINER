import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from transformers import AutoConfig
from transformers import AutoModelWithLMHead


class BertTagger(nn.Module):

    def __init__(self, output_dim, params):
        super(BertTagger, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.output_dim = output_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True # 输出每个token的表示
        config.output_attentions = True # 输出每个token的attention map
        self.encoder = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
  
        self.classifier = CosineLinear(self.hidden_dim, self.output_dim)
   

    def forward(self, X, return_feat=False):
        features = self.forward_encoder(X)
        logits = self.forward_classifier(features)
        if return_feat:
            return logits, features
        return logits
        
    def forward_encoder(self, X):
        '''
         features
        features[0]: hidden_states; 
        features[1]: all_hidden_states(embedding layer + 12 layer output hidden, 13*(bsz, seq_len, hidden_dim)); 
        features[2]: all_self_attentions(12 layer attention map, 12*(bsz, att_heads=12, seq_len, seq_len))
        '''
        features = self.encoder(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        features = features[1][-1] # (bsz, seq_len, hidden_dim)
        return features

    def forward_classifier(self, features, embed_mean=None):
        logits = self.classifier(features)
        return logits



class CosineLinear(nn.Module):
    def __init__(self, hidden_dim, output_dim, sigma=True):
        super(CosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, hidden_dim))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input, num_head=1):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        if num_head>1:
            out=[]
            head_dim = input.size(1)//num_head
            input_list = torch.split(input, head_dim, dim=1)
            input_list = [F.normalize(input_item, p=2,dim=1) for input_item in input_list]
            weight_list = torch.split(self.weight, head_dim, dim=1)
            weight_list = [F.normalize(weight_item, p=2,dim=1) for weight_item in weight_list]
            for n_input, n_weight in zip(input_list, weight_list):
                out.append(F.linear(n_input, n_weight))
            out = sum(out)
        else:
            out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        return out
    

class SplitCosineLinear(nn.Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, hidden_dim, old_output_dim, new_output_dim, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = old_output_dim + new_output_dim
        self.fc0 = CosineLinear(hidden_dim, 1, False) # for "O" class
        self.fc1 = CosineLinear(hidden_dim, old_output_dim-1, False)
        self.fc2 = CosineLinear(hidden_dim, new_output_dim, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, num_head=1):
        out0 = self.fc0(x, num_head=num_head)
        out1 = self.fc1(x, num_head=num_head)
        out2 = self.fc2(x, num_head=num_head)
        out = torch.cat((out0, out1, out2), dim=-1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out





