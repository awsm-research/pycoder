"""
Code completion for line-level prediction (evaluation only)
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataset import lineDataset
from beam import Beam

from fuzzywuzzy import fuzz

from transformers import (GPT2Config, GPT2Tokenizer, GPT2LMHeadModel)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}      

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens

def eval_line_completion(args, model, tokenizer, token_types, file_type='test'):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT") or to_add in token_types
            ):
                if not codes.endswith(" "):
                    codes += " " + to_add + " "
                else:
                    codes += to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    dataset = lineDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100, is_type=('type' in args.predict_mode), is_repeat_type=(args.predict_mode=='repeat_type'))
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    mrr_preds = []
    edit_sim = 0.0
    em = 0.0
    mrr = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 16
            m = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs[:, :-1])[1]
            p = []       
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in outputs]
                past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100): 
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    outputs = model(input_ids, past_key_values=past_hidden)

                    out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in outputs[1]]
                    past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]
                pred = [torch.cat([x.view(-1) for x in p]+[zero]*(100-len(p))).view(1,-1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).replace("<EOL>", "").strip() #.strip("<EOL>").strip() 
                else:
                    text = DecodeIds(t).strip("{").strip()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0

                ## MRR calculation ##
                if args.calculate_mrr:
                    mrr_pred = []
                    mrr_temp = 0
                    for mrr_idx in range(beam_size):
                        t = pred[mrr_idx].cpu().numpy()
                        t = t.tolist()
                        if 0 in t:
                            t = t[:t.index(0)]
                        if args.langs == "python":
                            text = DecodeIds(t).replace("<EOL>", "").strip() #.strip("<EOL>").strip() 
                        else:
                            text = DecodeIds(t).strip("{").strip()
                        if text == gt[0] and (1/(mrr_idx+1) > mrr_temp):
                            mrr_temp = 1/(mrr_idx+1)
                        mrr_pred.append(text)
                    mrr += mrr_temp
                    mrr_preds.append(mrr_pred)
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
    
    # pickle.dump(preds, open(os.path.join(args.output_dir, "line_preds.pkl"), "wb"))
    # pickle.dump(gts, open(os.path.join(args.output_dir, "line_gts.pkl"), "wb"))

    checkpoint_num = args.pretrain_dir.split("/")[-1]
    saved_file = os.path.join(args.output_dir, checkpoint_num + '_' + args.predict_mode + "_predictions_line.txt")
    with open(saved_file, "w") as f:
        for pred_text in preds:
            f.write(pred_text+"\n")
            
    logger.info(f"Test {len(preds)} samples, saved at {saved_file}")
    logger.info(f"Edit sim: {edit_sim/len(preds)}, EM: {em/len(preds)}")
    if args.calculate_mrr:
        pickle.dump(mrr_preds, open(os.path.join(args.output_dir, checkpoint_num + '_' + args.predict_mode + "_mrr_preds.pkl"), "wb"))
        logger.info(f"MRR: {mrr/len(preds)}")

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")
    parser.add_argument("--load_name", type=str, default="pretrained", 
                        help="Load pretrained model name")

    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_line", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--not_pretrain', action='store_true',
                        help="use different dataset")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="node index if multi-node running")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")  

    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)  

    ####################################
    ##### pycoder modify arguments #####
    ####################################

    parser.add_argument('--no_type_tokens', action='store_true')
    parser.add_argument('--predict_mode', type=str, default='code') # code, type, repeat_type
    parser.add_argument('--calculate_mrr', action='store_true')
    parser.add_argument('--dataset_mode', type=str, default='our') #our, original

    ####################################

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    logger.info("local_rank: %d, node_index: %d, gpu_per_node: %d"%(args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
                   torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)
    token_types = ['<NAME>', '<KEYWORD>', '<NUMBER>', '<STRING>', '<NEWLINE>', '<INDENT>', '<DEDENT>', '<LPAR>', '<RPAR>', '<LSQB>', '<RSQB>', '<COLON>', '<COMMA>', '<SEMI>', '<PLUS>', '<MINUS>', '<STAR>', '<SLASH>', '<VBAR>', '<AMPER>', '<LESS>', '<GREATER>', '<EQUAL>', '<DOT>', '<PERCENT>', '<LBRACE>', '<RBRACE>', '<EQEQUAL>', '<NOTEQUAL>', '<LESSEQUAL>', '<GREATEREQUAL>', '<TILDE>', '<CIRCUMFLEX>', '<LEFTSHIFT>', '<RIGHTSHIFT>', '<DOUBLESTAR>', '<PLUSEQUAL>', '<MINEQUAL>', '<STAREQUAL>', '<SLASHEQUAL>', '<PERCENTEQUAL>', '<AMPEREQUAL>', '<VBAREQUAL>', '<CIRCUMFLEXEQUAL>', '<LEFTSHIFTEQUAL>', '<RIGHTSHIFTEQUAL>', '<DOUBLESTAREQUAL>', '<DOUBLESLASH>', '<DOUBLESLASHEQUAL>', '<AT>', '<ATEQUAL>', '<RARROW>', '<ELLIPSIS>', '<ERRORTOKEN>']
    if args.no_type_tokens:
        token_types = []
    special_tokens.extend(token_types)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        model = model_class.from_pretrained(pretrained)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        config = config_class.from_pretrained(args.config_dir)
        model = model_class(config)
        model.resize_token_embeddings(len(tokenizer))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Only works on single GPU
    if args.eval_line:
        eval_line_completion(args, model, tokenizer, token_types, file_type="test")

if __name__ == "__main__":
    main()
