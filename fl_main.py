from FC import FC_model
import torch
import random
import os
from model import *

from Fed_utils import * 
from option import args_parser, modify_command_options


from dataloader import *

from utils import *
from torch.nn import functional as F

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(args)

    setup_seed(args.seed) 

    class_per_entity = len(args.schema)-1


    domain_name = args.data_root.split("/")[-1]

    ner_dataloader = NER_dataloader(data_path=args.data_root,
                                    domain_name=domain_name,
                                    batch_size=args.batch_size, 
                                    entity_list=args.entity_list,
                                    n_samples=args.n_samples,
                                    is_filter_O=args.is_filter_O,
                                    schema=args.schema,
                                    is_load_disjoin_train=args.is_load_disjoin_train,
                                    sample_ratio=args.sample_ratio,
                                    incremental_method=args.incremental_method
                                    )
    
    label_list = ner_dataloader.label_list
    # Initialize the model for the first group of types
    if args.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
        # BERT-based NER Tagger
        model_g = BertTagger(output_dim=(1+class_per_entity*args.FG), params=args)
        model_g_old = None
    else:
        raise Exception('model name %s is invalid'%args.model_name)
    
    num_clients = args.num_clients  # 10

    models = []
    for client_index in range(50): 
        model_temp = FC_model(client_index, args.batch_size, device, args.entropy_threshold,ner_dataloader)
        models.append(model_temp)


    old_step = -1

    for ep_g in range(args.epochs_global): 

        current_step = ep_g // args.steps_global 


        if current_step != old_step:  # 进入新step 
            
            best_f1 = -1

            if current_step==0:
                new_entity_list = ner_dataloader.entity_list[:args.FG]
                all_seen_entity_list = ner_dataloader.entity_list[:args.FG]
            else:
                new_entity_list = ner_dataloader.entity_list[\
                                args.FG+(current_step-1)*args.PG
                                :args.FG+current_step*args.PG]
                all_seen_entity_list = ner_dataloader.entity_list[\
                                :args.FG+current_step*args.PG]
                
            num_classes_new = 1+class_per_entity*len(all_seen_entity_list) # 截至当前任务的标签数量

            if current_step>0:
                num_classes_old = num_classes_new - class_per_entity*len(new_entity_list) #旧任务的标签数量
                old_classes = num_classes_old
            else:
                num_classes_old = 0 #第一个任务 旧标签数量为0
                old_classes = num_classes_old

            new_classes_list = list(range(num_classes_old,num_classes_new))
      
                
            dev_loader = ner_dataloader.get_dataloader(first_N_classes=-1,
                                                            select_entity_list=new_entity_list,
                                                            phase=['dev'],
                                                            is_filter_O=args.is_filter_O,
                                                            reserved_ratio=args.reserved_ratio)[0]

            test_loader = ner_dataloader.get_dataloader(first_N_classes=len(all_seen_entity_list),
                                                            select_entity_list=[],
                                                            phase=['test'],
                                                            is_filter_O=False)[0]
            
            
        if current_step != old_step and old_step != -1:  # 进入新step 且不是第0个
            args.base_weights = False
            
            for i in range(num_clients):
                models[i].last_entropy = -1
            
            num_clients = num_clients + args.add_clients

            if current_step == 1:
                model_g_old = deepcopy(model_g)

                hidden_dim = model_g.classifier.hidden_dim
                output_dim = model_g.classifier.output_dim
     
                print("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                        hidden_dim,
                                        output_dim,
                                        class_per_entity*args.PG))
                new_fc = SplitCosineLinear(hidden_dim, output_dim, class_per_entity*args.PG)
                new_fc.fc0.weight.data = model_g.classifier.weight.data[:1] # for O class
                new_fc.fc1.weight.data = model_g.classifier.weight.data[1:]
                new_fc.sigma.data = model_g.classifier.sigma.data

                model_g.classifier = new_fc

            elif current_step > 1:
                model_g_old = deepcopy(model_g)

                hidden_dim = model_g.classifier.hidden_dim
                output_dim1 = model_g.classifier.fc1.output_dim
                output_dim2 = model_g.classifier.fc2.output_dim
       
                print("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                                            hidden_dim,
                                                            1+output_dim1+output_dim2,
                                                            class_per_entity*args.PG))                                                
                new_fc = SplitCosineLinear(hidden_dim, 1+output_dim1+output_dim2, class_per_entity*args.PG)

                new_fc.fc0.weight.data = model_g.classifier.fc0.weight.data # for O classes
                new_fc.fc1.weight.data[:output_dim1] = model_g.classifier.fc1.weight.data
                new_fc.fc1.weight.data[output_dim1:] = model_g.classifier.fc2.weight.data
                new_fc.sigma.data = model_g.classifier.sigma.data

                model_g.classifier = new_fc

        if current_step != old_step:
            for i in range(num_clients):
                models[i].current_dev_loader = dev_loader
        
        print('federated global round: {}, step: {}'.format(ep_g, current_step))

        w_local = []

        local_clients = int(num_clients*args.local_clients_ratio)
        clients_index = random.sample(range(num_clients), local_clients) 

  
        print('select part of clients to conduct local training') 
        print(clients_index)

        for c in clients_index:
            local_model = local_train(args, models, c, model_g, model_g_old,current_step,new_entity_list,old_classes,new_classes_list)
            w_local.append(local_model)


        print('federated aggregation...')

        if args.base_weights == False:
            w_g_new = FedAvg(w_local)  
            model_g.load_state_dict(w_g_new) 
        
            f1_dev, _ = evaluate(model_g, dev_loader, device,label_list, entity_order=new_entity_list)
            
            #  选择在当前任务开发集上表现最好的模型
            if f1_dev > best_f1: # 默认是micro平均，这个是首选指标
                best_f1 = f1_dev

                if current_step==0:
                    # base model
                    save_model(model_g,f"{args.checkpoint_base}/{args.dataset}_{args.FG}_base.pth")
                else: # 其他任务 最好的模型
                    save_model(model_g,f"{args.checkpoint_path}/{args.dataset}_{args.task}_{args.incremental_method}_step_{current_step}.pth")
  
        
    
        if ((ep_g+1)% args.steps_global)==0: # 多个round中 最好的一个

            if current_step==0:
                base_ckpt_path = f"{args.checkpoint_base}/{args.dataset}_{args.FG}_base.pth"
            else:
                base_ckpt_path = f"{args.checkpoint_path}/{args.dataset}_{args.task}_{args.incremental_method}_step_{current_step}.pth"

            ckpt = torch.load(base_ckpt_path)

            model_g.hidden_dim = ckpt['hidden_dim']
            model_g.output_dim = ckpt['output_dim']
            model_g.encoder.load_state_dict(ckpt['encoder'])
            model_g.classifier = ckpt['classifier']
            

            f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = evaluate(model_g, test_loader, 
                                                device,label_list,
                                                each_class=True,
                                                entity_order=all_seen_entity_list) 


            print("Accumulation: Test_f1=%.3f, Test_ma_f1=%.3f, Test_f1_each_class=%s"%(
                        f1_test_cumul, ma_f1_test_cumul, str(f1_test_each_class_cumul)))
            print("Finish testing the %d-th iter!"%(current_step+1))            

               
        old_step = current_step



if __name__ == '__main__':
    
    args = args_parser() 
    args = modify_command_options(args)

    args.checkpoint_base = f"{args.checkpoint}"
    args.checkpoint_path = f"{args.checkpoint}/seed_{args.seed}"

    os.makedirs(args.checkpoint_path, exist_ok=True) 
    
    main(args)
