# This code is modify from CodeXGLUE from Microsoft.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import os
import pickle5 as pickle
import gc
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_langs_%s"%(args.langs)+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if 'train' in file_type:
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []
            if args.langs == 'all':
                langs = os.listdir(args.data_dir)
            else:
                langs = [args.langs]

            data=[]
            for lang in langs:
                datafile = os.path.join(args.data_dir, lang, file_type+'.pkl')
                if 'train' in file_type:
                    logger.warning("Creating features from dataset file at %s", datafile)
                dataset = pickle.load(open(datafile, 'rb'))
                data.extend(['<s> '+' '.join(x['function'].split())+' </s>' for idx,x in enumerate(dataset) if idx%world_size==local_rank])

            data = data
            length = len(data)
            logger.warning("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids)
            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])            
            del input_ids
            gc.collect()

            if 'train' in file_type:
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])

class finetuneDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if 'train' in file_type:
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            if 'train' in file_type:
                logger.warning("Creating features from dataset file at %s", datafile)
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids) // world_size
            logger.info(f"tokens: {length*world_size}")
            input_ids = input_ids[local_rank*length: (local_rank+1)*length]

            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])            
            del input_ids
            gc.collect()

            if 'train' in file_type:
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])

class EvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, token_types, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("load %d"%(percent))
            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(input_ids, tokenizer, logger, token_types, block_size=block_size)
            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def split(self, input_ids, tokenizer, logger, token_types, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    token = tokenizer.convert_ids_to_tokens(sample[block_size-1-j])
                    if token in token_types or token[0] == '\u0120' or token.startswith("<NUM_LIT"):
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]
            # print(len(sample))
            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])
        


class lineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='dev', block_size=924, is_type=False, is_repeat_type=False):
        if args.dataset_mode == 'original':
            file_type = 'original_' + file_type
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                (self.inputs, self.gts) = pickle.load(handle)
        else:
            path = '../../line-level/dataset/py150/line_completion/'
            datafile = os.path.join(path, f"{file_type}.json")
            with open(datafile) as f:
                datas = f.readlines()

            length = len(datas)
            logger.info("Data size: %d"%(length))
            self.inputs = []
            self.gts = []
            for data in tqdm(datas):
                data = json.loads(data.strip())
                self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
                self.gts.append(data["gt"])

            with open(cached_file, 'wb') as handle:
                    pickle.dump((self.inputs, self.gts), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]
