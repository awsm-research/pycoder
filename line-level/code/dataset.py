# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function
import os
import pickle
import gc
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset 

class lineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924, is_type=False, is_repeat_type=False):
        if is_type:
            if args.predict_mode == 'both' or is_repeat_type:
                file_type = 'repeat_' + file_type
            file_type = 'type_' + file_type
        if args.dataset_mode == 'original':
            file_type = 'original_' + file_type
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                (self.inputs, self.gts) = pickle.load(handle)
        else:        
            datafile = os.path.join(args.data_dir, f"{file_type}.json")
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
