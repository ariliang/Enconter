import argparse

parser = argparse.ArgumentParser(description="Train a transformer")
# Basic config
parser.add_argument("--epoch", type=int, default=10, help="epoch")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--save_dir", type=str, default="checkpoint", help="save directory")
parser.add_argument("--save_epoch", type=int, default=5, help="save per how many epoch")
# Optimizer
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--lr_override", action="store_true", help="ignore optimizer checkpoint and override learning rate")
parser.add_argument("--weight_decay", type=float, default=1, help="lr weight decay factor")
parser.add_argument("--decay_step", type=int, default=1, help="lr weight decay step size")
parser.add_argument("--warmup", action="store_true", help="Learning rate warmup")
parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup step")
# Dataset
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataset loader")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--dataset_version", type=str, help="dataset version")
# model
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Choose between bert_initialized or original")
parser.add_argument("--tokenizer", type=str, default='bert-base-cased', help="Using customized tokenizer")
# Debug
parser.add_argument("--no_shuffle", action="store_false", help="No shuffle")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--debug_dataset_size", type=int, default=1)

args = parser.parse_args()