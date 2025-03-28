import argparse


def modify_command_options(opts):

    if opts.dataset == 'i2b2':
        opts.num_types = (16 + 1)
    elif opts.dataset == 'ontonotes5':
        opts.num_types = (18+1)
    else:
        raise NotImplementedError(f"Unknown dataset: {opts.dataset}")
    
    opts.checkpoint = './checkpoints'
    FG, PG = opts.task.split('-')
    opts.FG = int(FG)
    opts.PG = int(PG)

    return opts

def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data/NER_data/", help="source domain")
    parser.add_argument('--dataset', type=str, default='i2b2', choices=['i2b2', 'ontonotes5'], help="name of dataset")
    parser.add_argument("--task", type=str, default="8-2", choices=['8-1', '8-2', '10-1', '10-2'], help="Task to be executed")
    parser.add_argument('--incremental_method', type=str, default='OURS', choices=['FT', 'PODNet', 'LUCIR', 'ST', 'ExtendNER', 'CFNER', 'CPFD', 'CPFD1', 'CPFD.5', 'CPFD-L', 'OURS.5', 'OURS','OURS1','OURS3','OURS-Con','OURS-Hid'], help="name of method")
    parser.add_argument('--sample_ratio', type=float, default=0.6, help='ratio of data for local clients (num_type=1)')
    parser.add_argument('--batch_size', type=int, default=16, help='size of mini-batch') # 统一双卡跑 每个卡是8  
    parser.add_argument('--epochs_local', type=int, default=10, help='local epochs of each global round')  # num_pg_type=1 5; num_pg_type>1 10
    parser.add_argument('--lr1', type=float, default=2e-3, help='learning rate step=0') # fixed no decay
    parser.add_argument('--lr2', type=float, default=4e-4, help='learning rate step>0') # fixed no decay
    parser.add_argument('--num_clients', type=int, default=10, help='initial number of clients') 
    parser.add_argument('--add_clients', type=int, default=4, help='the number of new add clients for each step') 
    parser.add_argument('--local_clients_ratio', type=float, default=0.4, help='the ratio of selected clients in each round')  # fixed
    parser.add_argument('--epochs_global', type=int, default=25, help='total number of global rounds') 
    parser.add_argument('--steps_global', type=int, default=5, help='the global rounds for each step (task)') 
    parser.add_argument("--seed", type=int, default=2024, help="random seed (default: 2024)")
    parser.add_argument("--use_entropy_detection", action="store_true", default=False) 
    parser.add_argument("--entropy_threshold", type=float, default=0.6, help="threshold for entropy detection")
    

    # Model
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base or wide_resnet)")
    parser.add_argument("--base_weights", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")
    parser.add_argument("--alpha", type=float, default=0, help="Trade-off parameter")
    parser.add_argument("--none_idx", type=int, default=103, help="None token index(103=[mask])")

    
    # Data
    parser.add_argument("--schema", type=str, default="BIO", choices=['IO','BIO','BIOES'], help="Lable schema")
    parser.add_argument("--entity_list", type=str, default="", help="entity list")
    parser.add_argument("--n_samples", type=int, default=-1, help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")
    parser.add_argument("--is_filter_O", default=False, help="If filter out samples contains only O labels")
    parser.add_argument("--is_load_disjoin_train", default=True, help="If loading the join ckpt for training dat (only for CL)")
    parser.add_argument("--reserved_ratio", type=float, default=0, help="the ratio of reserved samples")


    # Training Settings
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max length for each sentence") 
    parser.add_argument("--mu", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--extra_annotate_type", type=str, default='none', choices=['none','current','all'] , help="Simulate mannual annotation in each data split")
    parser.add_argument("--info_per_epochs", type=int, default=1, help="Print information every how many epochs")



    # Incremental Learning Settings
    # =========================================================================================
    parser.add_argument("--is_fix_trained_classifier", default=True, help="If fix the trained classifer")
    parser.add_argument("--is_unfix_O_classifier", default=False, help="If not fix the O classifer")
    

    # =========================================================================================
    # Baseline Settings
    # =========================================================================================
    # 1.Distillate / 2.Self-Training (add temperature or not)
    parser.add_argument("--is_distill", default=False, action='store_true', help="If using distillation model for baseline")
    parser.add_argument("--distill_weight", type=float, default=1, help="distillation weight for loss")
    parser.add_argument("--is_ranking_loss", default=False, action='store_true', help="Add ranking loss in LUCIR")
    parser.add_argument("--ranking_weight", type=float, default=5, help="weight for ranking loss")
    parser.add_argument("--adaptive_distill_weight", default=True, help="If using adaptive weight")
    parser.add_argument("--adaptive_schedule", type=str, default='root', choices=['root','linear','square'], help="The schedule for adaptive weight")
    parser.add_argument("--temperature", type=float, default=1, help="temperature of the student model")
    parser.add_argument("--ref_temperature", type=float, default=1, help="temperature of the teacher model")


    # 3.LUCIR
    parser.add_argument("--is_lucir", default=False, action='store_true', help="If using LUCIR as baseline")
    parser.add_argument("--lucir_lw_distill", type=float, default=50, help="Loss weight for distillation")
    parser.add_argument("--lucir_K", type=int, default=1, help="Top K for MR loss")
    parser.add_argument("--lucir_mr_dist", type=float, default=0.5, help="Margin for MR loss")
    parser.add_argument("--lucir_lw_mr", type=float, default=1, help="Loss weight for MR loss")

    # 4.PodNet
    parser.add_argument("--is_podnet", default=False, action='store_true', help="If using Podnet as baseline")
    parser.add_argument("--podnet_is_nca", default=False, action='store_true', help="If using NCA loss")
    parser.add_argument("--podnet_nca_scale", type=float, default=1, help="The scaling factor for NCA")
    parser.add_argument("--podnet_nca_margin", type=float, default=0.6, help="The margin for NCA")
    parser.add_argument("--podnet_lw_pod_flat", type=float, default=1, help="Loss weight for flatten (last) feature distillation loss")
    parser.add_argument("--podnet_lw_pod_spat", type=float, default=1, help="Loss weight for intermediate feature distillation loss")
    parser.add_argument("--podnet_normalize", default=False, action='store_true', help="If normalize the feature before calculating the distance")

    
    # =========================================================================================
    # DCE Settings
    # =========================================================================================
    parser.add_argument("--is_DCE", default=False, action='store_true', help="If using DCE")
    parser.add_argument("--is_ODCE", default=False, action='store_true', help="If using DCE for the predefined O classes")
    parser.add_argument("--top_k", type=int, default=3, help="Number of reference samples")


    ### CPFD
    parser.add_argument("--is_cpfd", default=False, action='store_true', help="If using CPFD as baseline")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--classif_adaptive_factor", default=True)
    parser.add_argument("--classif_adaptive_min_factor", default=0.0, type=float)


    ### OURS
    parser.add_argument("--is_ours", default=False, action='store_true')
    parser.add_argument("--hidd_fea_distill", default=False, action='store_true')
    parser.add_argument("--svd", default=False, action='store_true')
    parser.add_argument("--conloss_prototype", default=False, action='store_true')


    args = parser.parse_args()
    return args
