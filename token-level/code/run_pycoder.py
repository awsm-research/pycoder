"""
Code completion for token-level prediction (training and evaluation)
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle5 as pickle
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TextDataset, finetuneDataset, EvalDataset, lineDataset
from beam import Beam
from fuzzywuzzy import fuzz
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                        GPT2Config, GPT2Tokenizer, GPT2LMHeadModel)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

def load_and_cache_examples(args, tokenizer, evaluate=False, is_type=False, is_repeat_type=False):
    file_type = 'dev' if evaluate else 'train'
    if is_type:
        if args.train_mode == 'both' or is_repeat_type:
            file_type = 'repeat_' + file_type
            if 'adaptedGPT2' in args.pretrain_dir:
                file_type = 'adapt_' + file_type
        file_type = 'type_' + file_type
    if args.dataset_mode == 'original':
        file_type = 'original_' + file_type
    if args.not_pretrain:
        dataset = finetuneDataset(tokenizer, args, logger, file_type=file_type, 
                                block_size=args.block_size)
    else:
        dataset = TextDataset(tokenizer, args, logger, file_type=file_type,
                                block_size=args.block_size)
    return dataset         

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size

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

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def train(args, model, tokenizer, token_types, fh, pool, train_dataset=[], train_type_dataset=[], type_model=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
    
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.train_mode == 'code':
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    elif args.train_mode == 'type' or args.train_mode == 'repeat_type':
        train_type_sampler = RandomSampler(train_type_dataset)
        train_dataloader = DataLoader(train_type_dataset, sampler=train_type_sampler, batch_size=args.batch_size, drop_last=True)
    elif args.train_mode == 'both':
        train_dataloader = DataLoader(ConcatDataset(train_type_dataset, train_dataset), args.batch_size, drop_last=True, shuffle=True)


    total_examples = len(train_dataloader) * args.batch_size * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if type_model is not None:
        type_model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if type_model is not None:
        optimizer_grouped_parameters.extend([
        {'params': [p for n, p in type_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in type_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ])
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        logger.warning(f"Loading optimizer from {optimizer_last}")
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))   
    if args.local_rank == 0:
        torch.distributed.barrier()   
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples )
    logger.info("  Num epoch = %d", t_total*batch_size//total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0, 0.0, global_step
    outputs, outputs_type, soft_sharing_loss = None, None, None
    model.zero_grad()
    if type_model is not None:
        type_model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        for step, (batch) in enumerate(train_dataloader):
            if args.train_mode == 'both':
                batch_type = batch[0]
                batch = batch[1]
            elif args.train_mode == 'type' or args.train_mode == 'repeat_type':
                batch_type = batch
            
            if args.model_amount == 1:
                ########################################
                ###### PyCoder-Hard or PyCoder-IFN #####
                ########################################
                model.train()
                loss = 0.0
                w1 = 0.0
                if args.train_mode == 'code' or args.train_mode == 'both':
                    inputs, labels = (batch, batch)
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    outputs = model(inputs, labels=labels)
                    if args.train_mode == 'both' and args.loss_weight_value == -1:
                        w1 = torch.rand(1).uniform_(0.,1).to(args.device) #0.2
                        loss += outputs[0] * (1 - w1)
                    elif args.train_mode == 'both' and args.loss_weight_value != None:
                        loss += outputs[0] * args.loss_weight_value
                    else:
                        loss += outputs[0]
                if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
                    inputs_type, labels_type = (batch_type, batch_type)
                    inputs_type = inputs_type.to(args.device)
                    labels_type = labels_type.to(args.device)
                    outputs_type = model(inputs_type, labels=labels_type)
                    if args.train_mode == 'both' and args.loss_weight_value == -1:
                        loss += outputs_type[0] * w1
                    elif args.train_mode == 'both' and args.loss_weight_value != None:
                        loss += outputs_type[0] * (1 - args.loss_weight_value)
                    else:
                        loss += outputs_type[0]
            else:
                ########################################
                ############# PyCoder-Soft #############
                ########################################
                loss = 0.0   

                model.train()
                inputs, labels = (batch, batch)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                outputs = model(inputs, labels=labels)

                type_model.train()
                inputs_type, labels_type = (batch_type, batch_type)
                inputs_type = inputs_type.to(args.device)
                labels_type = labels_type.to(args.device)
                outputs_type = type_model(inputs_type, labels=labels_type)

                if args.loss_weight_mode == 'none':
                    # no weight loss #
                    loss = outputs[0] + outputs_type[0]
                elif args.loss_weight_mode == 'rlw':
                    # RLW weight #
                    weights = torch.nn.functional.softmax(torch.randn(3), dim=-1)
                    loss = outputs[0]*weights[0] + outputs_type[0]*weights[1]
                elif args.loss_weight_mode == 'rlw_ctr':
                    # RLW control weight #
                    w1 = torch.rand(1).uniform_(0.,0.2).to(args.device)
                    w2 = torch.rand(1).uniform_(0.2,0.5).to(args.device)
                    w3 = (1.0 - (w1 + w2)).to(args.device)
                    loss = outputs_type[0] * w2 + outputs[0] * w3
                elif args.loss_weight_mode == 'manual':
                    # manual #
                    loss = outputs[0]*0.7 + outputs_type[0]*0.2

                if args.soft_share:
                    soft_sharing_loss = 0.0
                    for param1, param2 in zip(model.parameters(), type_model.parameters()):
                        soft_sharing_loss += torch.norm(param1 - param2, p='fro')
                    
                    if args.loss_weight_mode == 'none':
                        # no weight loss #
                        loss += soft_sharing_loss
                    elif args.loss_weight_mode == 'rlw':
                        # RLW weight #
                        loss += soft_sharing_loss * weights[2]
                    elif args.loss_weight_mode == 'rlw_ctr':
                        # RLW control weight #
                        loss += soft_sharing_loss * w1
                    elif args.loss_weight_mode == 'manual':
                        # manual #
                        loss += soft_sharing_loss * 0.1

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if type_model is not None:
                    torch.nn.utils.clip_grad_norm_(type_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True

                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.logging_steps == 0:
                    logger.info("  steps: %s  ppl: %s  lr: %s", global_step, round(avg_loss,5), scheduler.get_last_lr()[0])
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, token_types, eval_when_training=True, type_model=type_model)
                        for key, value in results.items():
                            if value != None:
                                logger.info("  %s = %s", key, round(value,4))                    
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step, round(results['value_perplexity'],4)))
                        type_output_dir = os.path.join(args.output_dir, '{}-{}-{}-{}'.format('type', checkpoint_prefix, global_step, round(results['type_perplexity'],4)))
                    else:
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        type_output_dir = os.path.join(args.output_dir, "{}-{}-{}".format('type', checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if type_model is not None and not os.path.exists(type_output_dir):
                            os.makedirs(type_output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    if type_model is not None:
                        type_model_to_save = (
                            type_model.module if hasattr(type_model, "module") else type_model
                        )
                    model_to_save.save_pretrained(output_dir)
                    if type_model is not None:
                        type_model_to_save.save_pretrained(type_output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    if type_model is not None:
                        logger.info("Saving type_model checkpoint to %s", type_output_dir)
                    # _rotate_checkpoints(args, checkpoint_prefix)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    if type_model is not None:
                        type_last_output_dir = os.path.join(args.output_dir, 'type-checkpoint-last')
                        if not os.path.exists(type_last_output_dir):
                            os.makedirs(type_last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    if type_model is not None:
                        type_model_to_save.save_pretrained(type_last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')

                    # save loss history #
                    loss_history_file = os.path.join(last_output_dir, 'loss_history.csv')
                    if not os.path.exists(loss_history_file):
                        with open(loss_history_file, 'w', encoding='utf-8') as lossf:
                            lossf.write("global_step,loss_type,loss_value,loss_share,loss_avg,value_perplexity,type_perplexity,sum_perplexity,line_edit_similarity,line_exact_match\n")
                    with open(loss_history_file, 'a', encoding='utf-8') as lossf:
                        wvalue_perplexity = round(results['value_perplexity'],4) if args.evaluate_during_training else None
                        wtype_perplexity = round(results['type_perplexity'],4) if args.evaluate_during_training else None
                        wsum_perplexity = round(results['perplexity'],4) if args.evaluate_during_training else None
                        wedit_sim = results['edit_similarity'] if args.evaluate_during_training else None
                        wexact_match = results['exact_match'] if args.evaluate_during_training else None
                        wloss_value = outputs[0] if outputs is not None else None
                        wloss_type = outputs_type[0] if outputs_type is not None else None
                        lossf.write(f"{global_step},{wloss_value},{wloss_type},{soft_sharing_loss},{round(float(loss),4)},{wvalue_perplexity},{wtype_perplexity},{wsum_perplexity},{wedit_sim},{wexact_match}\n")
                    

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, token_types, prefix="", eval_when_training=False, type_model=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    if args.train_mode == 'code' or args.train_mode == 'both':
        eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
        eval_type_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, is_type=True, is_repeat_type=(args.train_mode=='repeat_type'))

    if args.train_mode == 'code':
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
    elif args.train_mode == 'type' or args.train_mode == 'repeat_type':
        eval_type_sampler = SequentialSampler(eval_type_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_type_dataset, sampler=eval_type_sampler, batch_size=args.eval_batch_size, drop_last=True)
    elif args.train_mode == 'both':
        eval_dataloader = DataLoader(ConcatDataset(eval_type_dataset, eval_dataset), args.eval_batch_size, drop_last=True, shuffle=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss, value_eval_loss, type_eval_loss = 0.0, 0.0, 0.0
    nb_eval_steps = 0
    value_perplexity, type_perplexity = 0.0, 0.0

    model.eval()
    if type_model is not None:
        type_model.eval()
       
    for batch in eval_dataloader:
        if args.train_mode == 'both':
            batch_type = batch[0]
            batch = batch[1]
        elif args.train_mode == 'type' or args.train_mode == 'repeat_type':
            batch_type = batch

        if args.train_mode == 'code' or args.train_mode == 'both':
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
        if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
            inputs_type, labels_type = (batch_type, batch_type)
            inputs_type = inputs_type.to(args.device)
            labels_type = labels_type.to(args.device)

        with torch.no_grad():
            lm_loss = 0.0
            type_loss = 0.0
            value_loss = 0.0
            if args.model_amount == 1:
                if args.train_mode == 'code' or args.train_mode == 'both':
                    outputs = model(inputs, labels=labels)
                    lm_loss += outputs[0]
                    value_loss = outputs[0]
                if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
                    outputs_type = model(inputs_type, labels=labels_type)
                    lm_loss += outputs_type[0]
                    type_loss = outputs_type[0]
            else:
                if args.train_mode == 'code' or args.train_mode == 'both':
                    outputs = model(inputs, labels=labels)
                    lm_loss += outputs[0]
                    value_loss = outputs[0]
                if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
                    outputs_type = type_model(inputs_type, labels=labels_type)
                    lm_loss += outputs_type[0]
                    type_loss = outputs_type[0]
            eval_loss += lm_loss.mean().item()
            if args.train_mode == 'code' or args.train_mode == 'both':
                value_eval_loss += value_loss.mean().item()
            if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
                type_eval_loss += type_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    if args.train_mode == 'code' or args.train_mode == 'both':
        value_eval_loss = value_eval_loss / nb_eval_steps
        value_perplexity = torch.exp(torch.tensor(value_eval_loss))
    if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
        type_eval_loss = type_eval_loss / nb_eval_steps
        type_perplexity = torch.exp(torch.tensor(type_eval_loss))
    
    es = 0.0 ; em = 0.0 
    if args.validate_line:
        es, em = eval_line_completion(args, model, tokenizer, token_types, file_type="dev") 

    result = {
        "perplexity": float(perplexity),
        "value_perplexity": float(value_perplexity),
        "type_perplexity": float(type_perplexity),
        "edit_similarity": float(es),
        "exact_match": float(em) * 100
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            #logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def eval_acc(args, model, tokenizer, token_types, file_type='test'):
    """
    Evaluate token level code completion on accuracy.

    This function can only used to evaluate accuracy, but not inference, because the inputs are previous sub-tokens but not tokens.
    But it can be guaranteed that the accuracy in this function is the same as the real token level completion.
    The reason is:
    Assuming the inputs are "context_len = 100 <EOL> masks = np . zeros (", and the ground truth is "context_len".
    Due to our bpe encoding, the model have to outputs "context", "_" and "len" in 3 time step, i.e. gt0="context", gt1="_", gt2="len".
    In a real inference scenario:
    time step 0, inputs "context_len = 100 <EOL> masks = np . zeros ( ", model outputs: out0;
    time step 1, inputs: in1=out0, outputs: out1
    ... until the model outputs a complete token
    But in this function, no matter out0 is, in1=gt0="context".
    That is to say, in this function, we feed ground truth but not output sub-token when we predict the next token which is split by bpe.
    So obviouly we would get different predictions from the real token completion scenario.
    However, if we calculate token leval accuracy, 
    if and only if the model predicts every sub-token correctly, the complete token can be seen correct.
    In this situation, out0==gt0, out1==gt1, so it doesn't matter we feed gt or output to model.
    In summary, this function can make models oupout the same complete token if this token equals to ground truth, 
    if not, the model might predict a different token from the real completion scenario, but all wrong.
    So it would not affect the token level accuracy.

    I use this trick to speed up evaluation due to the large test set.
    """
    if args.predict_mode == 'type':
        file_type = 'type_' + file_type
    elif args.predict_mode == 'repeat_type':
        if "adapt" in args.pretrain_dir:
            file_type = 'type_adapt_repeat_' + file_type
        else:
            file_type = 'type_repeat_' + file_type
    if args.dataset_mode == 'original':
        file_type = 'original_' + file_type
    eval_dataset = EvalDataset(tokenizer, args, logger, token_types, file_type=file_type, block_size=args.block_size)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node)

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
    

    model.eval()

    correct = 0.0
    total = 0

    total_pred = []
    total_gt = []
    raw_pred = []
    raw_gt = []

    for step, batch in enumerate(eval_dataloader):
        inputs = batch.to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            pred_scores = outputs[0]
            pred_ids = pred_scores.argmax(-1)
        all_pred = []
        all_gt = []
        prev_pred = None
        for pred, gt in zip(pred_ids, inputs):
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()
            raw_gt.extend(gt)
            raw_pred.extend(pred)
            for i, y in enumerate(gt):
                if i == 0:
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                        all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = []
                        now_pred = []
                    else:
                        now_gt = [y] 
                        now_pred = [0] if prev_pred is None else [prev_pred]
                else:
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                    if tokenizer.decode(y) in token_types or y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT"):
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = [y]
                        now_pred = [pred[i-1]]
                        try:
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append("<SPACE>")
                        all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = []
                        now_pred = []
                        continue
                    now_gt.append(y)
                    now_pred.append(pred[i-1])

        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        for x, y in zip(all_pred, all_gt):
            if y not in ["<s>","</s>","<EOL>", "<pad>", "<DEDENT>", "<INDENT>"]: #["<s>","</s>","<EOL>", "<pad>"]: 
                total += 1
                if x == y:
                    correct += 1
        
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
            logger.info(f"{total}, {correct/total}")

    # pickle.dump(total_pred, open(os.path.join(args.output_dir, "preds.pkl"), "wb"))
    # pickle.dump(total_gt, open(os.path.join(args.output_dir, "gts.pkl"), "wb"))
    # pickle.dump(raw_pred, open(os.path.join(args.output_dir, "raw_preds.pkl"), "wb"))
    # pickle.dump(raw_gt, open(os.path.join(args.output_dir, "raw_gts.pkl"), "wb"))

    if args.predict_mode == "type":
        file_predict_name = "type_predictions.txt"
    elif args.predict_mode == "repeat_type":
        file_predict_name = "repeat_type_predictions.txt"
    else:
        file_predict_name = "predictions.txt"
    # if args.dataset_mode == 'original':
    #     file_predict_name = 'original_' + file_predict_name
    checkpoint_num = args.pretrain_dir.split("/")[-1]
    saved_file = os.path.join(args.output_dir, checkpoint_num + "_" + file_predict_name)
    total_samples = post_process(args, total_pred, total_gt, open(os.path.join(args.data_dir, f"{file_type}.txt")).readlines(), saved_file)
    logger.info(f"Eval on {total_samples}, saved at {saved_file}")
    
    return total, correct

def post_process(args, preds, gts, true_gts, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []
    open_tag = False
    for i, (pred,gt) in enumerate(zip(preds,gts)):
        if gt in ["", "<pad>"]: #["", "<pad>","<s>","</s>"]:
            continue
        elif gt == "<s>":
            open_tag = True
        elif not open_tag and gt == "</s>":
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if open_tag and gt == "</s>": #"END" or "END" in gt:
            open_tag = False
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            true_gt_list = true_gts[cnt].strip().split()
            while true_gt_list.count('</s>') > 1:
                true_gt_list.pop()
            assert gt_str == ' '.join(true_gt_list), f"{cnt} sample gt_str != true_gt" #true_gts[cnt].strip()
            wf.write(pred_str+"\n")
            cnt += 1
            new_gt = []
            new_pred = []
    
    return cnt

def accuracy(pred_list, gt_list):
    correct = 0
    total = 0
    for i in range(len(gt_list)):
        for j in range(len(gt_list[i])):
            if gt_list[i][j] not in ["<s>","</s>","<EOL>", "<pad>", "<DEDENT>", "<INDENT>"]: #["<s>","</s>","<EOL>", "<pad>"]: 
                total += 1
                if gt_list[i][j] == pred_list[i][j]:
                    correct += 1
    # print(f'accuracy: {correct/total}')
    return correct, total

def eval_line_completion(args, model, tokenizer, token_types, file_type='dev'):
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

    dataset = lineDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    # model.zero_grad()
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
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
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
        # if step % args.logging_steps == 0:
        #     logger.info(f"{step} are done!")
        
    # logger.info(f"Test {len(preds)} samples")
    # logger.info(f"Edit sim: {edit_sim/len(preds)}, EM: {em/len(preds)}")
    return edit_sim/len(preds), em/len(preds)

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

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

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
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="node index if multi-node running")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")  
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)

    ####################################
    ##### pycoder modify arguments #####
    ####################################

    parser.add_argument('--no_type_tokens', action='store_true')
    parser.add_argument('--train_mode', type=str, default='code') # code, type, repeat_type, both
    parser.add_argument('--model_amount', type=int, default=1) # 1, 2
    parser.add_argument('--soft_share', action='store_true')
    parser.add_argument('--predict_mode', type=str, default='code') # code, type, repeat_type
    parser.add_argument('--loss_weight_mode', type=str, default='none') # none, rlw, uct, manual, rlw_crt
    parser.add_argument('--loss_weight_value', default=None, type=float) # -1 for random_ctr
    parser.add_argument('--validate_line', action='store_true') # validate line-level on dev set 
    parser.add_argument('--dataset_mode', type=str, default='our') #our, original

    ####################################
    
    pool = None
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "bigbird"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
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

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_dir = os.path.join(checkpoint_last, 'config.json')
        if args.model_amount > 1:
            type_checkpoint_last = os.path.join(args.output_dir, 'type-checkpoint-last')
            args.type_pretrain_dir = os.path.join(type_checkpoint_last)
            args.type_config_dir = os.path.join(type_checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} steps".format(checkpoint_last, args.start_step))

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
        type_model = None
        if args.model_amount > 1 and args.train_mode == 'both':
            if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
                type_model = model_class.from_pretrained(args.type_pretrain_dir)
            else:
                type_model = model_class.from_pretrained(pretrained)
            type_model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        config = config_class.from_pretrained(args.config_dir)
        model = model_class(config)
        model.resize_token_embeddings(len(tokenizer))
        type_model = None
        if args.model_amount > 1 and args.train_mode == 'both':
            if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
                type_model = model_class(args.type_config_dir)
            else:
                type_model = model_class(config)
            type_model.resize_token_embeddings(len(tokenizer))


    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.model_amount > 1 and args.train_mode == 'both':
        type_model_parameters = model.parameters()
        num_params = sum([np.prod(p.size()) for p in type_model_parameters])
        logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = []
        train_type_dataset = []
        if args.train_mode == 'code' or args.train_mode == 'both':
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        if args.train_mode == 'type' or args.train_mode == 'repeat_type' or args.train_mode == 'both':
            train_type_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, is_type=True, is_repeat_type=(args.train_mode=='repeat_type'))

        global_step, tr_loss = train(args, model, tokenizer, token_types, fh, pool, train_dataset=train_dataset, train_type_dataset=train_type_dataset, type_model=type_model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Only works on single GPU
    if args.do_eval:
        # dev_total, dev_cr = eval_acc(args, model, tokenizer, 'dev')
        # logger.info(f"Dev total tokens: {dev_total}, accuracy: {dev_cr/dev_total}")
        
        # if args.model_amount>1 and args.predict_mode in ['type','repeat_type']:
        #     test_total, test_cr = eval_acc(args, type_model, tokenizer, token_types, 'test')
        # else:
        test_total, test_cr = eval_acc(args, model, tokenizer, token_types, 'test')
        
        logger.info(f"Test total tokens: {test_total}, accuracy: {test_cr/test_total}")


if __name__ == "__main__":
    main()
