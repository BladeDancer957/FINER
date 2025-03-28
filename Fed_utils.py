import torch.nn as nn
import torch
import copy
import numpy as np

import random

from seqeval.metrics import f1_score # 序列标注评估工具

pad_token_label_id = nn.CrossEntropyLoss().ignore_index  # -100

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def local_train(args, clients, index, model_g, model_g_old, current_step, new_entity_list,old_classes,new_classes_list):
    
    clients[index].beforeTrain(args, current_step,new_entity_list)
    if args.base_weights == False:
        if args.use_entropy_detection==True:
            clients[index].update_entropy_signal(model_g)
        local_model = clients[index].train(args, model_g,model_g_old,old_classes,new_classes_list)
    else:
        local_model = None

    return local_model

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg




def evaluate(model_g, dataloader, device, label_list, each_class=False, entity_order=[]):
    
    tmp_model_g = copy.deepcopy(model_g)
    tmp_model_g = tmp_model_g.to(device)

    with torch.no_grad():

        tmp_model_g.eval()

        y_list = []
        x_list = []
        logits_list = []

        for x, y in dataloader: 
            x = x.to(device, dtype=torch.long) # (bs,sequence_length)
            y = y.to(device, dtype=torch.long) # (bs,sequence_length)
            all_features = tmp_model_g.encoder(x)
            logits = tmp_model_g.forward_classifier(all_features[1][-1])   # (bsz, seq_len, output_dim) 

            _logits = logits.view(-1, logits.shape[-1]).detach().cpu()
            logits_list.append(_logits)
            x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
            x_list.append(x) 
            y = y.view(y.size(0)*y.size(1)).detach().cpu()
            y_list.append(y)

        
        y_list = torch.cat(y_list)
        x_list = torch.cat(x_list)
        logits_list = torch.cat(logits_list)   
        pred_list = torch.argmax(logits_list, dim=-1)

    

        ### calcuate f1 score
        pred_line = []
        gold_line = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id: # !=-100
                pred_token = label_list[pred_index] # label索引转label
                gold_token = label_list[gold_index]
                # lines.append("w" + " " + pred_token + " " + gold_token)
                pred_line.append(pred_token) 
                gold_line.append(gold_token) 

        # Check whether the label set are the same,
        # ensure that the predict label set is the subset of the gold label set
        gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
        if set(gold_label_set)!=set(pred_label_set):
            O_label_set = []
            for e in pred_label_set:
                if e not in gold_label_set:
                    O_label_set.append(e)
            if len(O_label_set)>0:
                # map the predicted labels which are not seen in gold label set to 'O'
                for i, pred in enumerate(pred_line):
                    if pred in O_label_set:
                        pred_line[i] = 'O'

        # compute overall f1 score
        # micro f1 (default)
        f1 = f1_score([gold_line], [pred_line])*100
        # macro f1 (average of each class f1)
        ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
        if not each_class: # 不打印每个类别的f1
            return f1, ma_f1

        # compute f1 score for each class
        f1_list = f1_score([gold_line], [pred_line], average=None)
        f1_list = list(np.array(f1_list)*100)
        gold_entity_set = set()
        for l in gold_label_set:
            if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                gold_entity_set.add(l[2:])
        gold_entity_list = sorted(list(gold_entity_set))
        f1_score_dict = dict()
        for e, s in zip(gold_entity_list,f1_list):
            f1_score_dict[e] = round(s,2)
        # using the default order for f1_score_dict
        if entity_order==[]:
            return f1, ma_f1, f1_score_dict
        # using the pre-defined order for f1_score_dict
        assert set(entity_order)==set(gold_entity_list),\
            "gold_entity_list and entity_order has different entity set!"
        ordered_f1_score_dict = dict()
        for e in entity_order:
            ordered_f1_score_dict[e] = f1_score_dict[e]


        tmp_model_g = tmp_model_g.to('cpu')
        torch.cuda.empty_cache() 
        del tmp_model_g

        return f1, ma_f1, ordered_f1_score_dict
    



def save_model(model_g, path=''):
    """
    save the best model
    """
 
    torch.save({
        "hidden_dim": model_g.hidden_dim,
        "output_dim": model_g.output_dim,
        "encoder": model_g.encoder.state_dict(),
        "classifier": model_g.classifier
    }, path)




