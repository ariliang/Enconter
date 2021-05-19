class TrainArguments:

    # Basic config
    epoch = 10      # default=10, epoch
    batch_size = 8  # default=4, batch size
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
    dataset_version = 'CoNLL' # dataset version

    # model
    model = 'bert-base-uncased' # default=bert-base-uncased
    tokenizer = 'bert-base-cased' # default='bert-base-cased', Using customized tokenizer

    # Debug
    no_shuffle = None # action=store_false, No shuffle
    debug = None # action=store_true, Debug mode
    debug_dataset_size = 1 # default=1

    @staticmethod
    def get_args(opt):
        args = TrainArguments

        if opt == 'pointer_e':
            args.save_dir = 'pointer_e'
            args.dataset = 'CoNLL_pointer_e'

            args.workers = 1
        elif opt == 'greedy_enconter':
            args.save_dir = 'greedy_enconter'
            args.dataset = 'CoNLL_greedy_enconter'
            args.warmup = True
        elif opt == 'bbt_enconter':
            args.save_dir = 'bbt_enconter'
            args.dataset = 'CoNLL_bbt_enconter'
            args.warmup = True

        return args


class TestArguments:

    # Basic config
    batch_size = 4          # default=4, Batch size
    save_dir = 'checkpoint' # default="checkpoint", Save directory
    eval_dataset = None     # type=str, required=True
    output_file = None      # type=str, required=True

    # model
    model = 'bert-base-cased'       # default="bert-base-cased", Choose between bert_initialized or original
    tokenizer = 'bert-base-cased'   # default="bert-base-cased", Using customized tokenizer
    inference_mode = 'normal'       # default="normal", Select inference mode between normal and esai

    @staticmethod
    def get_args(opt):
        args = TestArguments

        if opt == 'pointer_e':
            args.save_dir = 'pointer_e'
            args.eval_dataset = './dataset/CoNLL_test'
            args.output_file = 'pointer_e'
        elif opt == 'pointer_e_esai':
            args.save_dir = 'pointer_e'
            args.eval_dataset = './dataset/CoNLL_test_esai'
            args.output_file = 'pointer_e_esai'
            args.inference_mode = 'esai'
        elif opt == 'greedy_enconter':
            args.save_dir = 'greedy_enconter'
            args.eval_dataset = './dataset/CoNLL_test'
            args.output_file = 'greedy_enconter'
        elif opt == 'greedy_enconter_esai':
            args.save_dir = 'greedy_enconter'
            args.eval_dataset = './dataset/CoNLL_test_esai'
            args.output_file = 'greedy_enconter_esai'
            args.inference_mode = 'esai'
        elif opt == 'bbt_enconter':
            args.save_dir = 'bbt_enconter'
            args.eval_dataset = './dataset/CoNLL_test'
            args.output_file = 'bbt_enconter'
        elif opt == 'bbt_enconter_esai':
            args.save_dir = 'bbt_enconter'
            args.eval_dataset = './dataset/CoNLL_test_esai'
            args.output_file = 'bbt_enconter_esai'
            args.inference_mode = 'esai'

        return args


train_args = TrainArguments.get_args('pointer_e')
test_args = TestArguments.get_args('pointer_e')