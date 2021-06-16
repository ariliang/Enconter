import logging
import os
import pickle as pk

import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

from config.args import test_args as args


device = torch.device("cuda")
logger = logging.getLogger(__name__)


# Tokenizer
tokenizer_path = os.path.join(args.save_dir, args.tokenizer)
if os.path.exists(tokenizer_path):
    logger.info("Loading saved tokenizer in {}...".format(tokenizer_path))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
elif os.path.exists(args.model):
    logger.info("Loading saved tokenizer in {}...".format(args.model))
    tokenizer = BertTokenizer.from_pretrained(args.model)
else:
    assert False

padding_token = tokenizer.pad_token_id

# Load model
counter = 0
counter_path = os.path.join(os.getcwd(), args.save_dir, "counter.txt")
if os.path.exists(counter_path):
    with open(counter_path, "r") as counter_file:
        counter = int(counter_file.read())
else:
    assert False

logger.info("Loading checkpoint...")
model = BertForMaskedLM.from_pretrained(args.model)
model.resize_token_embeddings(len(tokenizer))
model_path = os.path.join(os.getcwd(), args.save_dir, "model_" + str(counter - 1) + ".ckpt")
if counter > 0:
    if os.path.exists(model_path):
        logger.info("Loading weight from %s", model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        logger.info("Model check point not exist!")
        assert False
model = model.to(device)

with open(args.eval_dataset, "rb") as fin:
    eval_dataset = pk.load(fin)

result = []
gen_iter = []
noi_token_num = tokenizer.additional_special_tokens_ids[0]
for eval in tqdm(eval_dataset[:100]):
    if args.inference_mode == 'esai':
        e, gt, span_arr = eval
        span_arr = torch.tensor(span_arr)
    else:
        e, gt, tp = eval
    # if e[-2:] != [tokenizer.sep_token, tokenizer.sep_token]:
    #     continue

    generated = torch.tensor([tokenizer.convert_tokens_to_ids(tkn) for tkn in e], dtype=torch.long)
    token_type = torch.tensor([tp], dtype=torch.long)
    start_idx = (token_type == 0).sum()

    if args.inference_mode == 'esai':
        assert len(generated) == len(span_arr)
    gen_iter_counter = 0

    while len(generated) < 512:
        inputs = generated.unsqueeze(0).to(device)
        token_type = token_type.to(device)
        token_type = torch.cat((token_type, torch.ones(1, inputs.shape[1]-token_type.shape[1], dtype=torch.long).to(device)), dim=-1)
        outputs = model(inputs, token_type)
        top_k_prob, indices = torch.topk(torch.softmax(outputs[0], dim=-1), k=20, dim=-1)
        predicted = torch.multinomial(top_k_prob.squeeze(), 1)
        predicted = torch.gather(indices.squeeze(), -1, predicted).squeeze()
        g_len = generated.shape[0]
        if args.inference_mode == 'esai':
            # Prevent from insert after the last token
            predicted = predicted[:-1]
            # newly constructed span to be inserted into the span_arr
            inserted_span = torch.full(predicted.shape, -1, dtype=torch.long)
            # determine valid insertion place
            valid_mask = (span_arr[1:] == -1) | (span_arr[:-1] == -1) | (span_arr[1:] != span_arr[:-1])
            predicted[~valid_mask] = noi_token_num
            if (predicted == noi_token_num).all():
                break
            generated_seq = (torch.zeros(g_len * 2 - 1, dtype=torch.long)
                             .scatter(0, torch.arange(0, g_len * 2, step=2), generated)
                             .scatter(0, torch.arange(1, g_len * 2 - 1, step=2), predicted.cpu()))
            new_span_arr = (torch.zeros(g_len * 2 - 1, dtype=torch.long)
                            .scatter(0, torch.arange(0, g_len * 2, step=2), span_arr)
                            .scatter(0, torch.arange(1, g_len * 2 - 1, step=2), inserted_span))
            span_arr = new_span_arr[generated_seq != noi_token_num]
        else:
            if (predicted[start_idx:] == noi_token_num).all():
                break
            generated_seq = (torch.zeros(g_len * 2, dtype=torch.long)
                             .scatter(0, torch.arange(0, g_len * 2, step=2), generated)
                             .scatter(0, torch.arange(1, g_len * 2, step=2), predicted.cpu()))
        generated_seq = generated_seq[generated_seq != noi_token_num]
        inf_inp_text = tokenizer.decode(generated.tolist())
        inf_out_text = tokenizer.decode(generated_seq.tolist())
        generated = generated_seq.clone().detach()
        gen_iter_counter += 1
    generation = tokenizer.decode(generated.tolist())
    # print("Entities: ", e)
    # print("Generation: ", generation)
    # print("GT: ", gt)
    result.append((e, generation, gt))
    if e == generation.split():
        gen_iter.append(-1)
    else:
        gen_iter.append(gen_iter_counter)
with open(os.path.join("output/eval", args.output_file), "wb") as fout:
    pk.dump(result, fout)
with open(os.path.join("output/eval", args.output_file + "_gen_iter"), "wb") as fout:
    pk.dump(gen_iter, fout)
