class Argument:

    # Basic config
    epoch = 10      # default=10, epoch
    batch_size = 4  # default=4, batch size
    save_dir = 'checkpoint' # save directory
    save_epoch = 5  # default=5, save per how many epoch

    # Optimizer
    lr = 5e-5           # default=5e-5, learning rate
    lr_override = None  # action=store_true, ignore optimizer checkpoint and override learning rate
    weight_decay = 1    # default=1, lr weight decay factor
    decay_step = 1      # default=1, lr weight decay step size
    warmup = None       # action=store_true, Learning rate warmup
    warmup_steps = 4000 # default=4000, Warmup step

    # Dataset
    workers = 8         # default=8, number of workers for dataset loader
    dataset = None      # required=True, path to dataset
    dataset_version = None # dataset version

    # model
    model = 'bert-base-uncased' # default=bert-base-uncased
    tokenizer = 'bert-base-cased' # default='bert-base-cased', Using customized tokenizer

    # Debug
    no_shuffle = None # action=store_false, No shuffle
    debug = None # action=store_true, Debug mode
    debug_dataset_size = 1 # default=1


def get_args(opt):
    args = Argument

    if opt == 'pointer_e':
        args.batch_size = 8
        args.save_dir = 'pointer_e'
        args.epoch = 10
        args.dataset = 'CoNLL_pointer_e'
        args.dataset_version = 'CoNLL'
        args.save_epoch = 5

        args.workers = 1
    elif opt == 'greedy_enconter':
        pass
    elif opt == 'bbt_enconter':
        pass

    return args


args = get_args('pointer_e')