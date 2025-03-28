import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer

from dataloader import *
from utils import *
from option import args_parser, modify_command_options

params = args_parser() 
params = modify_command_options(params)
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index  # -100


class BaseTrainer(object):
    def __init__(self, params, model, model_old, optimizer,device,old_classes,label_list):
        self.params = params # 配置
        self.model = model 
        self.model = self.model.to(device)
            
        self.refer_model = model_old 
        if self.refer_model:
            self.refer_model = self.refer_model.to(device)
            for par in self.refer_model.parameters():
                par.requires_grad = False
            self.refer_model.eval()

        self.optimizer = optimizer
        self.device = device
        self.old_classes = old_classes

        self.label_list = label_list
        


    def batch_forward(self, inputs, match_id_batch=None, O_match_id_batch=None, max_seq_length=512):    
        # Compute features
        self.inputs = inputs
        self.all_features = self.model.encoder(self.inputs)
        self.features  = self.all_features[1][-1]
        # Compute logits
        self.logits = self.model.forward_classifier(self.all_features[1][-1])   

        if match_id_batch!=None and len(match_id_batch)==0:
            self.features_match = None
            self.logits_match = None
        elif match_id_batch!=None and len(match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert self.dataloader_train!=None, "dataloader_train is none!"
            assert self.pos_matrix.any(), "pos_matrix is none!"
            # compute the sentences related to the match samples
            match_pos_matrix_batch = torch.tensor(self.pos_matrix[match_id_batch]).view(-1,2)
            select_sentence_idx = match_pos_matrix_batch[:,0]
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            select_sentence_batch = []
            for idx in unique_sentence_idx:
                select_sentence_batch.append(self.dataloader_train.dataset.X[idx])
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.to(self.device)
            # compute match feature
            with torch.no_grad():
                self.model.eval()
                tmp_features_match_lst = []
                for _select_batch in select_batch.split(8):
                    tmp_features_match_lst.append(self.model.forward_encoder(_select_batch))
                tmp_features_match = torch.cat(tmp_features_match_lst, dim=0)
                features_match = torch.FloatTensor(len(match_id_batch),tmp_features_match.shape[-1])
                for i, pos in enumerate(match_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    features_match[i] = tmp_features_match[select_to_unique_map[i]][pos_j]
                self.features_match = features_match.to(self.device)
                self.logits_match = self.model.forward_classifier(self.features_match)
                self.model.train()             

        if O_match_id_batch!=None and len(O_match_id_batch)==0:
            self.O_features_match = None
            self.O_logits_match = None
        elif O_match_id_batch!=None and len(O_match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert self.dataloader_train!=None, "dataloader_train is none!"
            assert self.O_pos_matrix.any(), "O_pos_matrix is none!"
            # compute the sentences related to the match samples
            match_O_pos_matrix_batch = torch.tensor(self.O_pos_matrix[O_match_id_batch])
            select_sentence_idx = match_O_pos_matrix_batch[:,0]
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            select_sentence_batch = []
            for idx in unique_sentence_idx:
                select_sentence_batch.append(self.dataloader_train.dataset.X[idx])
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.to(self.device)
            # compute match feature
            with torch.no_grad():
                self.model.eval()
                tmp_O_features_match_lst = []
                for _select_batch in select_batch.split(8):
                    tmp_O_features_match_lst.append(self.model.forward_encoder(_select_batch))
                tmp_O_features_match = torch.cat(tmp_O_features_match_lst, dim=0)
                O_features_match = torch.FloatTensor(len(O_match_id_batch),tmp_O_features_match.shape[-1])
                for i, pos in enumerate(match_O_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    O_features_match[i] = tmp_O_features_match[select_to_unique_map[i]][pos_j]
                self.O_features_match = O_features_match.to(self.device)
                self.O_logits_match = self.model.forward_classifier(self.O_features_match)
                self.model.train()


    def batch_loss(self, labels):
        '''
            Cross-Entropy Loss
        '''
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long()) # bs*seq_len, out_dim 默认自动忽略-100 label （pad、cls、sep、第二子词对应的索引）
        self.loss = ce_loss
        return ce_loss.item() 
    
    def batch_loss_podnet(self, labels):
        '''
            NCA Loss/Cross-Entropy Loss + Distillation Loss(CosineEmbeddingLoss+L2_norm)
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim

        # Check input
        assert self.refer_model != None, "refer_model is none!"
        assert self.inputs != None, "inputs is none!"
        assert self.inputs.shape[:2] == labels.shape[:2], "inputs and labels are not matched!"
        assert self.logits != None, "logits is none!"
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) NCA loss
        lsc_mask = torch.logical_and(labels >= refer_dims,
                                     labels != pad_token_label_id).flatten()

        if torch.sum(lsc_mask.float()) == 0:
            lsc_loss = torch.tensor(0., requires_grad=True).to(self.device)
        elif self.params.podnet_is_nca:
            similarities = self.logits.view(-1, all_dims)[lsc_mask]
            targets = labels.flatten().long()[lsc_mask]
            margins = torch.zeros_like(similarities)
            margins[torch.arange(margins.shape[0]), targets] = self.params.podnet_nca_margin
            similarities = self.params.podnet_nca_scale * (similarities - self.params.podnet_nca_margin)

            similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

            disable_pos = torch.zeros_like(similarities)
            disable_pos[torch.arange(len(similarities)),
                        targets] = similarities[
                            torch.arange(len(similarities)), targets]

            numerator = similarities[torch.arange(similarities.shape[0]),
                                     targets]
            denominator = similarities - disable_pos

            losses = numerator - torch.log(torch.exp(denominator).sum(-1))

            lsc_loss = torch.mean(-losses)
        else:
            lsc_loss = nn.CrossEntropyLoss()(self.logits.view(
                -1, all_dims)[lsc_mask], labels.flatten().long()[lsc_mask])

        # (2) distill loss
        distill_mask = torch.logical_and(labels==0,
                                         labels != pad_token_label_id)
        with torch.no_grad():
            self.model.eval()
            all_hidden_features = self.model.encoder(self.inputs)[1]
            self.refer_model.eval()
            refer_all_hidden_features = self.refer_model.encoder(self.inputs)[1]
            refer_features = refer_all_hidden_features[-1]
            self.model.train()

        lw_pod_flat = self.params.podnet_lw_pod_flat * math.sqrt(refer_dims/(all_dims-refer_dims))
        pod_flat_loss = lw_pod_flat * nn.CosineEmbeddingLoss(reduction='mean')(
                            self.features[distill_mask].view(-1, self.model.hidden_dim),
                            refer_features[distill_mask].view(-1, self.model.hidden_dim),
                            torch.ones(distill_mask.nonzero().size(0)).to(self.device))

        pod_spatial_loss = torch.tensor(0., requires_grad=True).to(self.device)
        for i, (a, b) in enumerate(zip(all_hidden_features, refer_all_hidden_features)):
            # shape of (batch_size, sent_len, hidden_dims)
            assert a.shape == b.shape, (a.shape, b.shape)

            a, b = a[distill_mask], b[distill_mask]

            if self.params.podnet_normalize:
                a = F.normalize(a, dim=-1, p=2)
                b = F.normalize(b, dim=-1, p=2)

            a = a.sum(dim=-1).unsqueeze(-1)# (-1, 1)
            b = b.sum(dim=-1).unsqueeze(-1)# (-1, 1)

            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
            pod_spatial_loss += layer_loss

        pod_spatial_loss = self.params.podnet_lw_pod_spat * (pod_spatial_loss/len(all_hidden_features))

        self.loss = lsc_loss + pod_flat_loss + pod_spatial_loss

        return lsc_loss.item(), pod_flat_loss.item() + pod_spatial_loss.item()
    
    def batch_loss_lucir(self, labels):
        '''
            Cross-Entropy Loss + Distillation Loss(CosineEmbeddingLoss) + MarginRankingLoss 
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim

        # Check input
        assert self.refer_model != None, "refer_model is none!"
        assert self.inputs != None, "inputs is none!"
        assert self.inputs.shape[:2] == labels.shape[:2], "inputs and labels are not matched!"
        assert self.logits != None, "logits is none!"
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) CE loss
        ce_mask = torch.logical_and(labels>=refer_dims,labels!=pad_token_label_id)
        if torch.sum(ce_mask.float()) == 0:
            ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:
            ce_loss = self.compute_CE(labels, ce_mask)

        # (2) distill loss
        lw_distill = self.params.lucir_lw_distill*math.sqrt(refer_dims/(all_dims-refer_dims))
        distill_mask = torch.logical_and(labels == 0, labels!=pad_token_label_id)
        # compute refer_features from refer_model
        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
        if torch.sum(distill_mask.float()) == 0:
            distill_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:
            distill_loss = lw_distill * nn.CosineEmbeddingLoss()(
                self.features[distill_mask].view(-1, self.model.hidden_dim),
                refer_features[distill_mask].view(-1, self.model.hidden_dim),
                torch.ones(distill_mask.nonzero().size(0)).to(self.device)
            )

        # (3) MR loss
        # 旧类别+O类别，无replay时只有O类别，同distill_mask
        mr_mask = torch.logical_and(labels<refer_dims,
                                    labels!=pad_token_label_id).flatten()
        labels_masked = labels.flatten().long()[mr_mask].view(-1, 1)
        mr_logits = self.logits.view(-1, self.logits.shape[-1])[mr_mask]
        gt_scores = mr_logits.gather(1, labels_masked).repeat(1, self.params.lucir_K)
        max_novel_scores = mr_logits[:, refer_dims:].topk(self.params.lucir_K, dim=1)[0]

        count = gt_scores.size(0)
        if count > 0:
            mr_loss = nn.MarginRankingLoss(margin=self.params.lucir_mr_dist)(gt_scores.view(-1), \
                max_novel_scores.view(-1), torch.ones(count*self.params.lucir_K).to(self.device)) * self.params.lucir_lw_mr
        else:
            mr_loss = torch.tensor(0., requires_grad=True).to(self.device)

        self.loss = ce_loss + distill_loss + mr_loss

        return ce_loss.item(), distill_loss.item()+mr_loss.item()
    
    def compute_CE(self, labels, ce_mask):
        '''
            Cross-Entropy Loss
        '''
        all_dims = self.logits.shape[-1]
        ce_loss = nn.CrossEntropyLoss()(self.logits[ce_mask].view(-1, all_dims),
                                labels[ce_mask].flatten().long())
        return ce_loss
    
    def compute_DCE(self, labels, ce_mask):
        '''
            DCE for labeled samples
        '''
        assert torch.sum(ce_mask.float()) == int(self.logits_match.shape[0]/self.params.top_k) \
                and self.logits_match.shape[0]%self.params.top_k == 0, \
                "length of ce_mask and the number of match samples are not equal!!!"
        # joint ce_loss
        logits_prob = F.softmax(self.logits.view(-1,self.logits.shape[-1]), dim=-1)
        logits_prob_match = F.softmax(self.logits_match, dim=-1)
        # print(logits_prob_match)
        logits_prob_match = torch.mean(logits_prob_match.reshape(-1, self.params.top_k, logits_prob_match.size(-1)), dim=1)
        # print(logits_prob_match)
        # print(logits_prob[ce_mask.flatten()])
        logits_prob_joint = (logits_prob[ce_mask.flatten()]+logits_prob_match)/2

        ce_loss = F.nll_loss(torch.log(logits_prob_joint+1e-10), labels[ce_mask])

        return ce_loss
    
    def compute_KLDiv(self, refer_logits, distill_mask):
        '''
            KLDivLoss
        '''
        refer_dims = refer_logits.shape[-1]

        # 1.log(distribution)
        old_class_score = F.log_softmax(
                            self.logits[distill_mask]/self.params.temperature,
                            dim=-1)[:,:refer_dims].view(-1, refer_dims)
        # 2.ref_distribution
        ref_old_class_score = F.softmax(
                            refer_logits[distill_mask]/self.params.ref_temperature, 
                            dim=-1).view(-1, refer_dims)

        distill_loss = nn.KLDivLoss(reduction='batchmean')(old_class_score, ref_old_class_score)

        return distill_loss
    
    def compute_ODCE(self, refer_logits, distill_mask):
        '''
            DCE for O samples
        '''
        refer_dims = refer_logits.shape[-1]
        assert self.O_pos_matrix_batch.any(), "O_pos_matrix_batch is none"

        # get mask for defined O samples
        defined_O_mask = torch.zeros(refer_logits.shape[:2]).cuda()
        defined_O_mask[self.O_pos_matrix_batch[:,0], self.O_pos_matrix_batch[:,1]] = 1
        defined_O_mask = torch.logical_and(distill_mask, defined_O_mask)
        assert torch.sum(defined_O_mask.float()) == int(self.O_logits_match.shape[0]/self.params.top_k) \
                and self.O_logits_match.shape[0]%self.params.top_k == 0, \
                "length of defined_O_mask and the number of 'O' match samples are not equal!!!"
        
        # get average scores of the matched samples
        # truncate the refer_dims before softmax to mitigate the imblance 
        # between old and new classes 
        O_logits_prob_match = F.softmax(
                            self.O_logits_match[:,:refer_dims]/self.params.temperature, 
                            dim=-1)
        O_logits_prob_match = torch.mean(O_logits_prob_match.view(-1, self.params.top_k, O_logits_prob_match.shape[-1]), dim=1)
        # get scores of the original samples
        old_class_score_all = F.softmax(
                            self.logits/self.params.temperature,
                            dim=-1)[:,:,:refer_dims]
        joint_old_class_score_all = old_class_score_all.clone()



        joint_old_class_score_all[defined_O_mask] = (old_class_score_all[defined_O_mask]+O_logits_prob_match)/2
        joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
        # 2.ref_distribution
        refer_logits[defined_O_mask] /= 1e-10 # Equals to applying CE to defined O samples, others is KLDivLoss
        ref_old_class_score = F.softmax(
                            refer_logits[distill_mask]/self.params.ref_temperature, 
                            dim=-1).view(-1, refer_dims)
        # KL divergence
        distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
        
        return distill_loss
    
    def batch_loss_distill(self, labels):
        '''
            Cross-Entropy Loss + Distillation loss(KLDivLoss)
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim
            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) CE loss
        ce_mask = torch.logical_and(labels>=refer_dims,labels!=pad_token_label_id) 
        if torch.sum(ce_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
        elif self.params.is_DCE and self.logits_match!=None:
            ce_loss = self.compute_DCE(labels, ce_mask)
        else:
            ce_loss = self.compute_CE(labels, ce_mask)

        # (2) Ditsillation loss
        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
            refer_logits = self.refer_model.forward_classifier(refer_features)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
        
        distill_mask = torch.logical_and(labels==0,labels!=pad_token_label_id)
        if torch.sum(distill_mask.float())==0:
            distill_loss = torch.tensor(0., requires_grad=True).to(self.device)
        elif self.params.is_ODCE and self.O_logits_match!=None:
            distill_loss = self.compute_ODCE(refer_logits, distill_mask)
        else:   
            distill_loss = self.compute_KLDiv(refer_logits, distill_mask)

        if not self.params.adaptive_distill_weight:
            distill_weight = self.params.distill_weight
        elif self.params.adaptive_schedule=='root':
            distill_weight = self.params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),0.5)
        elif self.params.adaptive_schedule=='linear':
            distill_weight = self.params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),1)
        elif self.params.adaptive_schedule=='square':
            distill_weight = self.params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),2)
        else:
            raise Exception('Invalid %s'%(self.params.adaptive_schedule))

        # (3) Ranking Loss
        if self.params.is_ranking_loss:
            mr_mask = torch.logical_and(labels<refer_dims,
                                        labels!=pad_token_label_id).flatten()
            labels_masked = labels.flatten().long()[mr_mask].view(-1, 1)
            mr_logits = self.logits.view(-1, self.logits.shape[-1])[mr_mask]
            gt_scores = mr_logits.gather(1, labels_masked).repeat(1, self.params.lucir_K)
            max_novel_scores = mr_logits[:, refer_dims:].topk(self.params.lucir_K, dim=1)[0]

            count = gt_scores.size(0)
            if count > 0:
                mr_loss = nn.MarginRankingLoss(margin=params.lucir_mr_dist)(gt_scores.view(-1), \
                    max_novel_scores.view(-1), torch.ones(count*params.lucir_K).to(self.device)) * self.params.lucir_lw_mr
            else:
                mr_loss = torch.tensor(0., requires_grad=True).to(self.device)
        
        # weighted sum
        if self.params.is_ranking_loss:
            self.loss = ce_loss + distill_weight*distill_loss + self.params.ranking_weight*mr_loss
            return ce_loss.item(), distill_weight*distill_loss.item() + self.params.ranking_weight*mr_loss.item()
        else:
            self.loss = ce_loss + distill_weight*distill_loss
            return ce_loss.item(), distill_weight*distill_loss.item()
        
    
    def select_O_samples(self, refer_flatten_feat_O_train, O_pos_matrix):
        assert refer_flatten_feat_O_train.shape[0]==O_pos_matrix.shape[0],\
            "refer_flatten_feat_O_train.shape[0]!=O_pos_matrix.shape[0] !!!"
        assert self.refer_model!=None, "refer_model is none !!!"
        # get the logits of the refer model
        with torch.no_grad():
            self.refer_model.eval()
            O_logits = []
            for O_feature in refer_flatten_feat_O_train.split(64):
                O_feature = O_feature.to(self.device)
                O_logits.append(self.refer_model.classifier(O_feature).cpu())        
        O_logits = torch.cat(O_logits, dim=0)
        O_predicts = torch.argmax(O_logits, dim=-1)

        select_mask = torch.not_equal(O_predicts, self.label_list.index('O'))
       

        print('Ratio of select samples %.2f%% (%d/%d).'%(\
                torch.sum(select_mask).item()/select_mask.shape[0]*100,
                torch.sum(select_mask).item(),
                select_mask.shape[0]
            )
        )

        return refer_flatten_feat_O_train[select_mask], O_pos_matrix[select_mask]


    def find_median(self, train_loader):
        
        bg_entropy_values_total = []
        bg_pseudo_labels_total = []
        for X, labels in train_loader: # 928*8*30
            X = X.to(self.device, dtype=torch.long) # (bs,sequence_length)
            labels = labels.to(self.device, dtype=torch.long) # (bs,sequence_length)

            with torch.no_grad():
                self.refer_model.eval()
                refer_features = self.refer_model.forward_encoder(X)
                refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)


            mask_bg = labels == 0  # (bsz, seq_len)  找出背景位置
            probas = torch.softmax(refer_logits, dim=-1)  # (bsz,seq_len,refer_dims)

            _, pseudo_labels = probas.max(dim=-1) # 最大值 以及 最大值所在位置

            bg_entropy_values = entropy(probas)[mask_bg].view(-1)
            bg_entropy_values_total.extend(bg_entropy_values.detach().cpu().numpy().tolist())

            bg_pseudo_labels = pseudo_labels[mask_bg].view(-1) # bsz*seq_len
            bg_pseudo_labels_total.extend(bg_pseudo_labels.detach().cpu().numpy().tolist())

    
        bg_entropy_values_total = np.array(bg_entropy_values_total,dtype=np.float32)
        bg_pseudo_labels_total = np.array(bg_pseudo_labels_total, dtype=np.int32)
        thresholds = np.zeros(self.old_classes, dtype=np.float32) #old_classes
        base_threshold = self.params.threshold #0.001
        for c in range(len(thresholds)):
            thresholds[c] = np.median(bg_entropy_values_total[bg_pseudo_labels_total==c])
            thresholds[c] = max(thresholds[c], base_threshold)

        return torch.from_numpy(thresholds).to(self.device)


    def before(self, train_loader):
        self.thresholds = self.find_median(train_loader)

    def calculate_sample_weight(self, labels): # labels (bsz, seq_len)
        background = labels == 0
        old_token = (labels < self.old_classes) & (labels != pad_token_label_id)
        old_token = old_token & (~background)
        new_token = labels >= self.old_classes
        old_token = torch.sum(old_token, 1)
        new_token = torch.sum(new_token, 1)
        new_token[new_token==0] = 1 # 某个样本没有新label 防止除0异常
        weight = 0.5 + F.sigmoid(old_token/new_token)
        return weight


    def batch_loss_cpfd(self, labels):
     
        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model 输出维度
        all_dims = self.model.classifier.output_dim
            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  

        with torch.no_grad():
            self.refer_model.eval()
            refer_all_features= self.refer_model.encoder(self.inputs)
            refer_features = refer_all_features[1][-1]
            refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
        
            refer_all_attention_features = refer_all_features[2]
        

        classif_adaptive_factor = 1.0
        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置

  
        probs = torch.softmax(refer_logits, dim=-1) # (bsz,seq_len,refer_dims)
        _, pseudo_labels = probs.max(dim=-1) # 最大概率 以及 最大概率所在位置

        mask_valid_pseudo = entropy(probs) < self.thresholds[pseudo_labels] # (bsz, seq_len)

        # All old labels that are NOT confident enough to be used as pseudo labels:
        labels[~mask_valid_pseudo & mask_background] = pad_token_label_id

        # All old labels that are confident enough to be used as pseudo labels:
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                    mask_background]
 
        if self.params.classif_adaptive_factor:
            # Number of old/bg tokens that are certain
            num = (mask_valid_pseudo & mask_background).float().sum(dim=-1)
            # Number of old/bg tokens
            den =  mask_background.float().sum(dim=-1)
            # If all old/bg tokens are certain the factor is 1 (loss not changed)
            # Else the factor is < 1, i.e. the loss is reduced to avoid
            # giving too much importance to new tokens
            classif_adaptive_factor = num / (den + 1e-6)
            classif_adaptive_factor = classif_adaptive_factor[:, None]

            if self.params.classif_adaptive_min_factor:
                classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.params.classif_adaptive_min_factor)

   
        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels) # 0 新类 旧类伪标签 -100(计算的loss为0)    (bsz,seq_len)
        loss = classif_adaptive_factor * loss

        # type balance

        pre_sample_weights = self.calculate_sample_weight(labels)
        sample_weights = torch.ones(loss.size()).to(self.device) 
        for i in range(pre_sample_weights.size(0)): 
            sample_weights[i][(labels[i] > 0) & (labels[i] < self.old_classes)] = pre_sample_weights[i] # 样本中 被伪标注出来的旧类token 赋予 计算的权重， 其他（新类，真正的背景）token对应权重1

  
        loss = sample_weights * loss


        ignore_mask = (labels!=pad_token_label_id) 
            
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar
        

        all_attention_features = self.all_features[2]
        
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id) # other class token (non-entity)

        if torch.sum(distill_mask.float())==0:
            distill_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:   
            # distill logits loss
            old_logits_score = F.log_softmax(
                                self.logits[distill_mask]/self.params.temperature,
                                dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
     
            ref_old_logits_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)

            distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)

            # distill attention feature loss
            '''
                all_attention_features(12 layer attention map, 12*(bsz, att_heads=12, seq_len, seq_len))
            '''
            distill_attention_features_loss = torch.tensor(0., requires_grad=True).to(self.device)
            for attention_features, refer_attention_features in zip(all_attention_features, refer_all_attention_features):
                assert attention_features.shape == refer_attention_features.shape, (attention_features.shape, refer_attention_features.shape)  # (bsz, heads=12, seq_len, seq_len)
                
                bsz, heads, seq_len, seq_len = attention_features.shape

                attention_features = torch.where(attention_features <= -1e2, torch.zeros_like(attention_features).to(self.device),
                                              attention_features)
                refer_attention_features = torch.where(refer_attention_features <= -1e2, torch.zeros_like(refer_attention_features).to(self.device),
                                              refer_attention_features)

                # pooled feature distillation
                pfd1 = torch.mean(attention_features, dim=1)
                rpfd1 = torch.mean(refer_attention_features, dim=1)

                pfd2 = torch.mean(attention_features, dim=2)
                rpfd2 = torch.mean(refer_attention_features, dim=2)

                pfd3 = torch.mean(attention_features, dim=3)
                rpfd3 = torch.mean(refer_attention_features, dim=3)

                layer_loss1 = nn.MSELoss(reduction='mean')(pfd1,rpfd1)
                layer_loss2 = nn.MSELoss(reduction='mean')(pfd2,rpfd2)
                layer_loss3 = nn.MSELoss(reduction='mean')(pfd3,rpfd3)

                layer_loss = layer_loss1 + layer_loss2 + layer_loss3

                distill_attention_features_loss += layer_loss

            distill_attention_features_loss = distill_attention_features_loss/len(all_attention_features)


            distill_loss = self.params.distill_weight*(distill_logits_loss + distill_attention_features_loss)
        

        if not self.params.adaptive_distill_weight: 
            distill_loss_coefficient = 1 
        elif self.params.adaptive_schedule=='root': 
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),0.5) 
        elif self.params.adaptive_schedule=='linear':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),1)
        elif self.params.adaptive_schedule=='square':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),2)
        else:
            raise Exception('Invalid %s'%(self.params.adaptive_schedule))

        self.loss = ce_loss + distill_loss_coefficient*distill_loss # 总loss

        return ce_loss.item(), distill_loss_coefficient*distill_loss.item()



    def batch_loss_ours(self, labels):
     
        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim # old model 输出维度
        all_dims = self.model.classifier.output_dim
            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  

        with torch.no_grad():
            self.refer_model.eval()
            refer_all_features= self.refer_model.encoder(self.inputs)
            refer_features = refer_all_features[1][-1]
            refer_logits = self.refer_model.forward_classifier(refer_features)# (bsz,seq_len,refer_dims)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"

            refer_all_hidden_features = refer_all_features[1]
        

        classif_adaptive_factor = 1.0
        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置

  
        probs = torch.softmax(refer_logits, dim=-1) # (bsz,seq_len,refer_dims)
        _, pseudo_labels = probs.max(dim=-1) # 最大概率 以及 最大概率所在位置

        mask_valid_pseudo = entropy(probs) < self.thresholds[pseudo_labels] # (bsz, seq_len)

        # All old labels that are NOT confident enough to be used as pseudo labels:
        labels[~mask_valid_pseudo & mask_background] = pad_token_label_id

        # All old labels that are confident enough to be used as pseudo labels:
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                    mask_background]
 
        if self.params.classif_adaptive_factor:
            # Number of old/bg tokens that are certain
            num = (mask_valid_pseudo & mask_background).float().sum(dim=-1)
            # Number of old/bg tokens
            den =  mask_background.float().sum(dim=-1)
            # If all old/bg tokens are certain the factor is 1 (loss not changed)
            # Else the factor is < 1, i.e. the loss is reduced to avoid
            # giving too much importance to new tokens
            classif_adaptive_factor = num / (den + 1e-6)
            classif_adaptive_factor = classif_adaptive_factor[:, None]

            if self.params.classif_adaptive_min_factor:
                classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.params.classif_adaptive_min_factor)

   
        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels) # 0 新类 旧类伪标签 -100(计算的loss为0)    (bsz,seq_len)
        loss = classif_adaptive_factor * loss


        ignore_mask = (labels!=pad_token_label_id) 
            
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar


        contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
        if self.params.conloss_prototype:
            cploss_label = Conloss_proposal(self.params)
            contrast_loss = cploss_label(self.features, refer_features, labels) /100
        

        all_hidden_features = self.all_features[1]
        
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id) # other class token (non-entity)

        if torch.sum(distill_mask.float())==0:
            distill_loss = torch.tensor(0., requires_grad=True).to(self.device)
        else:   
            # distill logits loss
            old_logits_score = F.log_softmax(
                                self.logits[distill_mask]/self.params.temperature,
                                dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)
     
            ref_old_logits_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)

            distill_logits_loss = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)
      

            # distill_hidden_loss
            '''
                all_hidden_features(12 layer  12*(bsz, seq_len, 768))
            '''
            distill_hidden_loss = torch.tensor(0., requires_grad=True).to(self.device)
            if self.params.hidd_fea_distill:
                for hidden_features, refer_hidden_features in zip(all_hidden_features, refer_all_hidden_features):

                    assert hidden_features.shape == refer_hidden_features.shape, (hidden_features.shape, refer_hidden_features.shape)  # (bsz, seq_len, 768)
                    
                    bsz, seq_len, hidden_dim = hidden_features.shape

                    hidden_features, refer_hidden_features = hidden_features[distill_mask], refer_hidden_features[distill_mask]

                    hidden_features = hidden_features.view(-1,hidden_dim) # bsz*seq_len(select_out),768
                    refer_hidden_features = refer_hidden_features.view(-1,hidden_dim)

                    if self.params.svd:
                        length = hidden_features.shape[0]
                        hidden_features = hidden_features.view(length,12,64).permute(1,0,2)

                        refer_hidden_features = refer_hidden_features.view(length,12,64).permute(1,0,2)  # (12,length,64)

                        hidden_features_svd,_,_ = torch.linalg.svd(hidden_feature,full_matrices=False) # full_matrices=False (12,length,min(length,64)) 提高效率
                        refer_hidden_features_svd,_,_ = torch.linalg.svd(refer_hidden_features,full_matrices=False) 

                        distill_hidden_loss += nn.MSELoss(reduction='mean')(hidden_features_svd,refer_hidden_features_svd)
                    else:
                        distill_hidden_loss += nn.MSELoss(reduction='mean')(hidden_features,refer_hidden_features)


                distill_hidden_loss /= len(all_hidden_features)

            distill_loss = self.params.distill_weight*(distill_logits_loss + distill_hidden_loss+ contrast_loss)
        

        if not self.params.adaptive_distill_weight: 
            distill_loss_coefficient = 1 
        elif self.params.adaptive_schedule=='root': 
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),0.5) 
        elif self.params.adaptive_schedule=='linear':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),1)
        elif self.params.adaptive_schedule=='square':
            distill_loss_coefficient = np.power((refer_dims-1)/(all_dims-refer_dims),2)
        else:
            raise Exception('Invalid %s'%(self.params.adaptive_schedule))

        self.loss = ce_loss + distill_loss_coefficient*distill_loss # 总loss

        return ce_loss.item(), distill_loss_coefficient*distill_loss.item()
            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()  
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

 