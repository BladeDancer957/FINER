import torch.nn as nn
import torch
from Fed_utils import * 

from trainer import *

from utils import *



def entropy(probabilities):

    entropy = -probabilities * torch.log(probabilities + 1e-8)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


class FC_model:

    def __init__(self, client_index, batch_size, device, entropy_threshold,ner_dataloader):

        super(FC_model, self).__init__()

        self.client_index = client_index 
        self.batch_size = batch_size

        self.old_model = None

        self.learned_step = -1

        self.signal = False 

        self.current_trainloader = None  

        self.device = device


        self.last_entropy = -1

        self.trainer_state = None

        self.entropy_threshold = entropy_threshold

        self.ner_dataloader = ner_dataloader

        self.current_dev_loader = None

    # get incremental train data
    def beforeTrain(self, args, current_step, new_entity_list):

        print("Current Client Index: ", self.client_index)

        if current_step != self.learned_step: 
            self.learned_step = current_step 

            self.current_trainloader = self.ner_dataloader.get_dataloader(
                                                            first_N_classes=-1,
                                                            select_entity_list=new_entity_list,
                                                            phase=['train'],
                                                            is_filter_O=args.is_filter_O,
                                                            reserved_ratio=args.reserved_ratio)[0]

            if args.use_entropy_detection==False:
                self.signal = True # rule

        
    def update_entropy_signal(self, model_g):
        
        
        tmp_model = copy.deepcopy(model_g)
        
        tmp_model = tmp_model.to(self.device)
        tmp_model.eval()

        self.signal = self.entropy_signal(tmp_model,self.current_trainloader)

        tmp_model = tmp_model.to('cpu')

        torch.cuda.empty_cache() 

        del tmp_model


    # train model
    def train(self,args, model_g, model_g_old,old_classes,new_classes_list): 

        model = copy.deepcopy(model_g)

        if self.learned_step  == 0:
            optimizer = torch.optim.SGD(model.parameters(),
                                            lr=args.lr1,
                                            momentum=args.mu,
                                            weight_decay=args.weight_decay)
        else:
            if args.is_fix_trained_classifier:
                # if fix the O classifier
                if args.is_unfix_O_classifier:
                    ignored_params = list(map(id, model.classifier.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(args.lr2),
                                'weight_decay': float(args.weight_decay)}, \
                                {'params': model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
                else:
                    ignored_params = list(map(id, model.classifier.fc1.parameters())) + \
                                    list(map(id, model.classifier.fc0.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(args.lr2),
                                'weight_decay': float(args.weight_decay)}, \
                                {'params': model.classifier.fc0.parameters(), 'lr': 0., 
                                'weight_decay': 0.}, \
                                {'params': model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
            else:
                tg_params = [{'params': model.parameters(), 'lr': float(args.lr2), 
                            'weight_decay': float(args.weight_decay)}]
                
            optimizer = torch.optim.SGD(tg_params, 
                                                momentum=args.mu)
           
        if self.signal:
            self.old_model = model_g_old

        if self.old_model is not None:
            model_old = copy.deepcopy(self.old_model)
        else:
            model_old = None


        trainer = BaseTrainer(
                args,
                model,
                model_old,
                optimizer = optimizer,
                device = self.device,
                old_classes = old_classes,
                label_list = self.ner_dataloader.label_list
                )
        trainer.pad_token_id = self.ner_dataloader.auto_tokenizer.pad_token_id

        if model_old:
            if 'CPFD' in args.incremental_method or 'OURS' in args.incremental_method:
                trainer.before(train_loader=self.current_trainloader)

             # Compute match samples for DCE
            if args.is_DCE or args.is_ODCE:
                (refer_flatten_feat_train, refer_flatten_feat_O_train) \
                    = compute_feature_by_dataloader(dataloader=self.current_trainloader,
                                                    feature_model=trainer.refer_model.encoder, 
                                                    select_label_groups=[
                                                        new_classes_list,
                                                        [self.ner_dataloader.O_index],
                                                    ],
                                                    is_normalize=True)
                
                trainer.dataloader_train = self.current_trainloader
                num_sentence_all = len(trainer.dataloader_train.dataset.y)
                
                # 1.2 flatten label list and compute the neighbor for each sample
                if args.is_DCE:
                    flatten_label_train, pos_matrix = get_flatten_for_nested_list(
                                                    trainer.dataloader_train.dataset.y, 
                                                    select_labels=new_classes_list,
                                                    is_return_pos_matrix=True,
                                                    max_seq_length=args.max_seq_length)
                    trainer.pos_matrix = pos_matrix
                    num_samples_all = len(flatten_label_train)
                    assert refer_flatten_feat_train.shape[0] == num_samples_all, \
                            "refer_flatten_feat_train.shape[0]!=num_samples_all !!!"
                    # compute the neighbor for each sample
                    match_id = get_match_id(refer_flatten_feat_train, args.top_k)
                    # save the space
                    del refer_flatten_feat_train

                if args.is_ODCE:
                    _, O_pos_matrix = get_flatten_for_nested_list(
                                                    trainer.dataloader_train.dataset.y, 
                                                    select_labels=[self.ner_dataloader.O_index],
                                                    is_return_pos_matrix=True)
                    
                    refer_flatten_feat_O_train, O_pos_matrix = trainer.select_O_samples(
                                                    refer_flatten_feat_O_train, 
                                                    O_pos_matrix)
                    if len(O_pos_matrix)>0:
                        trainer.O_pos_matrix = O_pos_matrix
                        num_O_samples_all = refer_flatten_feat_O_train.shape[0]
                        # compute the neighbor for each sample
                        O_match_id = get_match_id(refer_flatten_feat_O_train, args.top_k)
                        # save the space
                        del refer_flatten_feat_O_train
                    else:
                        trainer.O_pos_matrix = []
                        num_O_samples_all = 0
                        O_match_id = []

            

        best_f1 = -1

        for e in range(1, args.epochs_local+1):
            

            print("============== epoch %d ==============" % e)

            # loss list 总loss 蒸馏loss 交叉熵loss
            loss_list, distill_list, ce_list = [], [], []
            # average loss
            mean_loss = 0.0

            # sample count for DCE
            sample_id, O_sample_id, sentence_id = 0, 0, 0
      

            for X, y in self.current_trainloader:
             

                X = X.to(self.device, dtype=torch.long) # (bs,sequence_length)
                y = y.to(self.device, dtype=torch.long) # (bs,sequence_length)

                match_id_batch, O_match_id_batch = None, None

                # Use DCE
                if model_old and args.is_DCE:
                    batch_sent_ids = list(range(sentence_id,sentence_id+X.shape[0]))
                    # count the number of entities (not O) in the batch
                    num_samples_batch = np.count_nonzero(np.isin(pos_matrix[:,0],batch_sent_ids))
                    # get the reference feature and the match reference feature
                    match_id_batch = match_id[sample_id*args.top_k:(sample_id+num_samples_batch)*args.top_k]
                    # update count number
                    sample_id += num_samples_batch

                # Use ODCE
                if model_old and args.is_ODCE and len(O_match_id)>0:
                    batch_sent_ids = list(range(sentence_id,sentence_id+X.shape[0]))
                    # count the number of O sampls in the batch
                    num_O_sample_batch = np.count_nonzero(np.isin(O_pos_matrix[:,0],batch_sent_ids))
                    # compute the O_pos_matrix_batch
                    O_pos_matrix_batch = O_pos_matrix[np.isin(O_pos_matrix[:,0],batch_sent_ids)]
                    O_pos_matrix_batch[:,0] = O_pos_matrix_batch[:,0]-sentence_id
                    trainer.O_pos_matrix_batch = O_pos_matrix_batch
                    # get the reference feature and the match reference feature
                    O_match_id_batch = O_match_id[O_sample_id*args.top_k:(O_sample_id+num_O_sample_batch)*args.top_k]
                    # update count number
                    O_sample_id += num_O_sample_batch
        
                # Forward
                trainer.batch_forward(X, 
                                    match_id_batch=match_id_batch,
                                    O_match_id_batch=O_match_id_batch,
                                    max_seq_length=args.max_seq_length)
                
  
                # Compute loss
                if model_old:
                    if args.is_distill:
                        ce_loss, distill_loss = trainer.batch_loss_distill(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif args.is_lucir:
                        ce_loss, distill_loss = trainer.batch_loss_lucir(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif args.is_podnet:
                        ce_loss, distill_loss = trainer.batch_loss_podnet(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif args.is_cpfd:
                        ce_loss, distill_loss = trainer.batch_loss_cpfd(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif args.is_ours:
                        ce_loss, distill_loss = trainer.batch_loss_ours(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    else:
                        ce_loss = trainer.batch_loss(y)
                        ce_list.append(ce_loss) # 每个batch的loss
                else: #  第一个任务 只有ce loss
                    ce_loss = trainer.batch_loss(y)
                    ce_list.append(ce_loss) # 每个batch的loss

                total_loss = trainer.batch_backward() # 总loss
                loss_list.append(total_loss) # 追加每个batch的总loss
                mean_loss = np.mean(loss_list) # 平均每个batch的总loss
                mean_distill_loss = np.mean(distill_list) if len(distill_list)>0 else 0 # 平均每个batch的distill loss
                mean_ce_loss = np.mean(ce_list) if len(ce_list)>0 else 0 # 平均每个batch的ce loss

                # Update sentence count
                sentence_id += X.shape[0]
           
            # Check whether mismatching exists
            if model_old and args.is_DCE:
                assert sample_id==num_samples_all, "The sample_id and num_samples_all mismatch!"
                assert sentence_id==num_sentence_all, "The sentence_id and num_sentence_all mismatch!"
                if args.is_ODCE and len(O_match_id)>0:
                    assert O_sample_id==num_O_samples_all, "The O_sample_id and num_O_samples_all mismatch!"


            # Print training information
            if args.info_per_epochs>0 and e%args.info_per_epochs==0: # args.info_per_epochs=1    每隔一个epoch 输出信息s
                
                print("Epoch %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f"%(
                            e, mean_loss, \
                            mean_ce_loss, mean_distill_loss
                    ))
                
                trainer.model = trainer.model.to('cpu')
                f1_dev, _ = evaluate(trainer.model, self.current_dev_loader, self.device, self.ner_dataloader.label_list)
            
                #  选择在当前任务开发集上表现最好的模型
                if f1_dev > best_f1: # 默认是micro平均，这个是首选指标
                    best_f1 = f1_dev
                    local_model = trainer.model.state_dict() 

                trainer.model = trainer.model.to(self.device)
     

        trainer.model = trainer.model.to('cpu')
        torch.cuda.empty_cache() 

        if args.use_entropy_detection==False:
            self.signal = False   # rule

        del model
        del trainer.model
        del optimizer
        del trainer.optimizer

        if model_old:
            trainer.refer_model = trainer.refer_model.to('cpu')
            torch.cuda.empty_cache() 
            del model_old
            del trainer.refer_model

        del trainer

        return local_model

        

    def entropy_signal(self, tmp_model, loader):
        start_ent = True
        res = False

        for (X, _) in loader:
          
            X = X.to(self.device, dtype=torch.long) # (bs,sequence_length)
          
            with torch.no_grad():
                outputs = tmp_model(X) #(batch,sequence_length,classes)

              
            softmax_out = nn.Softmax(dim=-1)(outputs) #(batch,sequence_length,classes)
            ent = entropy(softmax_out) #(batch,sequence_length)

            if start_ent: 
                all_ent = ent.float().cpu().view(-1,1)

                start_ent = False
            else: 
                all_ent = torch.cat((all_ent, ent.float().cpu().view(-1,1)), 0)  # (b+,sequence_length)

        overall_avg = torch.mean(all_ent.squeeze(-1)).item()

        if overall_avg - self.last_entropy > self.entropy_threshold:  
            res = True
        
        self.last_entropy = overall_avg


        return res



