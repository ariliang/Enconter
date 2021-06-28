import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data


from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from dataset_utils import InsertionTransformerDataset, concat_fn
from utils import get_linear_schedule_with_warmup, get_lr
from config.args import train_args as args

# torch.distributed.init_process_group(backend='', rank=0, world_size=2)

device = torch.device(args.device)
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visiable

if not args.debug:
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
logger.info("Args...", vars(args))

# Tokenizer
tokenizer_path = os.path.join(args.save_dir, args.tokenizer)
if os.path.exists(tokenizer_path):
    logger.info("Loading saved tokenizer in {}...".format(tokenizer_path))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
else:
    logger.info("Using {} tokenizer...".format(args.model))
    tokenizer = BertTokenizer.from_pretrained(args.model)
    if args.dataset_version == "CoNLL":
        tokenizer.add_special_tokens({"additional_special_tokens": ["[NOI]", "\n"]})
    elif args.dataset_version == 'dialo':
        tokenizer.add_special_tokens({'additional_special_tokens': ['[NOI]', '\n', '[BOS]', '[EOS]', '[PAT]', '[DOC]']})
        tokenizer.bos_token = '[BOS]'
        tokenizer.eos_token = '[EOS]'
    else:
        raise ValueError("dataset/tokenizer config error!")
    os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)

# Building model
padding_token = tokenizer.pad_token_id
logger.info("Building model...")
model = BertForMaskedLM.from_pretrained(args.model)
model.resize_token_embeddings(len(tokenizer))
model = torch.nn.DataParallel(model)
model = model.to(device)

# Read model counter which records the training epoch of the current model
counter = 0
counter_path = os.path.join(os.getcwd(), args.save_dir, "counter.txt")
if not args.debug:
    if os.path.exists(counter_path):
        with open(counter_path, "r") as counter_file:
            counter = int(counter_file.read())
    else:
        with open(counter_path, "w") as counter_file:
            counter_file.write(str(counter))

# Loss history
loss_history_path = os.path.join(os.getcwd(), args.save_dir, "loss_history.npy")
if os.path.exists(loss_history_path):
    with open(loss_history_path, "r") as counter_file:
        loss_history = np.load(loss_history_path)
else:
    loss_history = np.zeros(shape=0)

# Load check points and set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
model_path = os.path.join(os.getcwd(), args.save_dir, "model_" + str(counter - 1) + ".ckpt")
optim_path = os.path.join(os.getcwd(), args.save_dir, "optim_" + str(counter - 1) + ".ckpt")

if counter > 0 and not args.debug:
    if os.path.exists(model_path):
        logger.info("Loading weight from %s", model_path)
        model.module.load_state_dict(torch.load(model_path))
    else:
        logger.info("Model check point not exist!")
    if args.lr_override:
        logger.info("Learning rate OVERRIDE!")
    elif os.path.exists(optim_path):
        logger.info("Loading optim from %s", optim_path)
        optimizer.load_state_dict(torch.load(optim_path))
    else:
        logger.info("Optimizer check point not exist!")
optimizer.param_groups[0]['initial_lr'] = optimizer.param_groups[0]['lr']

training_dataset = InsertionTransformerDataset(tokenizer, os.path.join(os.getcwd(), args.dataset), args.max_len)
if args.debug or args.workers == 1:
    loader = data.DataLoader(training_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=concat_fn)
else:
    loader = data.DataLoader(training_dataset,
                             batch_size=args.batch_size,
                             shuffle=args.no_shuffle,
                             collate_fn=concat_fn,
                             num_workers=args.workers)

# Setup scheduler
if len(training_dataset) % args.batch_size == 0:
    total_step = len(training_dataset) // args.batch_size
else:
    total_step = len(training_dataset) // args.batch_size + 1
step = counter * len(loader)
if args.warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_step * args.epoch,
                                                step if step != 0 else -1)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')

def compute_loss(logits, labels, token_type, criterion):
        mask = torch.logical_and(labels, token_type)

        logits_view = torch.argmax(logits, dim=-1)

        logits_view = logits_view * mask
        labels_view = labels * mask

        # logits_view = logits_view.view(-1, logits_view.size(-1)).float()
        # labels_view = labels_view.view(-1).long()
        # loss = criterion(logits_view, labels_view) / len(target_range)

        # return loss/args.batch_size, accuracy/args.batch_size

logger.info("Start training...")
epoch_loss = np.zeros(0)
for e in range(counter, args.epoch):
    pbar = tqdm(total=total_step, desc=f'batch {e+1}/{args.epoch}')
    avg_loss = np.zeros(shape=(1))
    for batch_num, batch_data in enumerate(loader):
        model.train()
        pbar.update(1)
        inputs, labels, token_type = batch_data

        # encoded_labels = labels.tolist()
        # encoded_inputs = inputs.tolist()
        # decoded = []
        # for ipt, lb in zip(encoded_inputs, encoded_labels):
        #     decoded.append(tokenizer.decode(ipt))
        #     decoded.append(tokenizer.decode(lb))

        inputs, labels, token_type = inputs.to(device), labels.to(device), token_type.to(device)
        attn_mask = (inputs != tokenizer.pad_token_id).float().to(device)
        position_ids = torch.arange(args.max_len, dtype=torch.long).to(args.device).unsqueeze(0).repeat_interleave(args.batch_size, dim=0)

        output = model(input_ids=inputs, attention_mask=attn_mask, token_type_ids=token_type, position_ids=position_ids)

        # loss, accuracy = compute_loss(output[0], labels, token_type, criterion)
        loss = output[0].mean()

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        if args.warmup:
            scheduler.step()

        # loss /= args.gredient_accumulation

        # loss.backward(loss.data)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # if ((batch_num+1) % args.gredient_accumulation) == 0:
            # print(f'update gredient | loss {loss}')
            # optimizer.zero_grad()
            # optimizer.step()
            # if args.warmup:
                # scheduler.step()
        avg_loss += loss.item()
        logger.info(f"Epoch: {e} lr: {get_lr(optimizer)} Avg NLLLoss: {avg_loss / (batch_num + 1)}")
        if args.debug and batch_num and batch_num % args.debug_dataset_size == 0:
            break
    if not args.warmup:
        scheduler.step()
    pbar.close()
    loss_history = np.concatenate((loss_history, avg_loss / len(loader)))
    np.save(os.path.join(os.getcwd(), args.save_dir, "loss_history"), loss_history)
    plt.plot(np.arange(1, len(loss_history)+1), loss_history, '-o')
    plt.title("loss history")
    plt.savefig(os.path.join(args.save_dir, "loss_history.png"))
    if not args.debug and (e % args.save_epoch == 0 or e == args.epoch - 1):
        torch.save(model.module.state_dict(), os.path.join(os.getcwd(), args.save_dir, "model_" + str(e) + ".ckpt"))
        torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), args.save_dir, "optim_" + str(e) + ".ckpt"))
        with open(counter_path, "w") as counter_file:
            counter_file.write(str(e + 1))
