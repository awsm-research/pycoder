from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
import json
import torch
import os
import pickle
from fuzzywuzzy import fuzz

#######################################
###### Data and Model Directory #######
#######################################

line_input_dir = '../dataset/py150/line_completion/'
data_dir = '../../token-level/save/<model_path>/'
model_dir = data_dir + '<checkpoint>'
output_dir = 'decoding_preds/hardshare_sampling_20/'
cached_dir = data_dir + 'test_blocksize_924'
lit_dir = '../dataset/py150/literals.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################
########### Configuration #############
#######################################

no_token_types = False # baseline:True

no_repeat_ngram_size = 0
num_return_sequences = 1
early_stopping = True
seed_number = 64

beam_size = 5
temperature = 0.5 # 1 for default
k_size = 3 # 0 for sampline, >0 for Top-K
p_size = 0.2 # for Top-p

method_list = [
            # 'greedy', 
            'beam', 
            # 'sampling', 
            # 'sampling_temp', 
            # 'top_k', 
            # 'top_p'
            ]

#######################################
#######################################

torch.manual_seed(seed_number)
print(output_dir)
print(f"Config:\nmethods:{method_list}\nBeam_size={beam_size}, temperature={temperature}, k_size={k_size}, p_size={p_size},\nno_token_types={no_token_types}, no_repeat_ngram_size={no_repeat_ngram_size}, seed_number={seed_number}\n")

#### Load Model ####

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
special_tokens = get_special_tokens(lit_dir)
token_types = ['<NAME>', '<KEYWORD>', '<NUMBER>', '<STRING>', '<NEWLINE>', '<INDENT>', '<DEDENT>', '<LPAR>', '<RPAR>', '<LSQB>', '<RSQB>', '<COLON>', '<COMMA>', '<SEMI>', '<PLUS>', '<MINUS>', '<STAR>', '<SLASH>', '<VBAR>', '<AMPER>', '<LESS>', '<GREATER>', '<EQUAL>', '<DOT>', '<PERCENT>', '<LBRACE>', '<RBRACE>', '<EQEQUAL>', '<NOTEQUAL>', '<LESSEQUAL>', '<GREATEREQUAL>', '<TILDE>', '<CIRCUMFLEX>', '<LEFTSHIFT>', '<RIGHTSHIFT>', '<DOUBLESTAR>', '<PLUSEQUAL>', '<MINEQUAL>', '<STAREQUAL>', '<SLASHEQUAL>', '<PERCENTEQUAL>', '<AMPEREQUAL>', '<VBAREQUAL>', '<CIRCUMFLEXEQUAL>', '<LEFTSHIFTEQUAL>', '<RIGHTSHIFTEQUAL>', '<DOUBLESTAREQUAL>', '<DOUBLESLASH>', '<DOUBLESLASHEQUAL>', '<AT>', '<ATEQUAL>', '<RARROW>', '<ELLIPSIS>', '<ERRORTOKEN>']
if no_token_types:
    token_types = []
special_tokens.extend(token_types)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, do_lower_case=False, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()
print('model loaded.')

#### Load Dataset ####

class lineDataset(Dataset):
    def __init__(self, tokenizer, file_type='test', block_size=924):
        if os.path.exists(cached_dir):
            with open(cached_dir, 'rb') as handle:
                (self.inputs, self.gts) = pickle.load(handle)
        else:
            datafile = os.path.join(line_input_dir, f"{file_type}.json")
            with open(datafile) as f:
                datas = f.readlines()
            self.inputs = []
            self.gts = []
            for data in tqdm(datas):
                data = json.loads(data.strip())
                self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
                self.gts.append(data["gt"])

            with open(cached_dir, 'wb') as handle:
                pickle.dump((self.inputs, self.gts), handle, protocol=pickle.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]

dataset = lineDataset(tokenizer, file_type='test', block_size=924)
test_sampler = SequentialSampler(dataset)
test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
print('line-dataset loaded.')

#### Untils ####

def strip_to_line_completion_output(output_id, input_id, tokenizer):
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
    generate_output = output_id[len(input_id):]
    try:
        sep_idx = (generate_output == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
    except IndexError:
        sep_idx = len(generate_output)
    t = generate_output[:sep_idx].tolist()
    if 0 in t:
        t = t[:t.index(0)]
    text = DecodeIds(t).replace("<EOL>", "").strip()
    # tokenizer.decode(generate_output[:sep_idx]).strip()
    return text

def set_up_metric_dict(metric_dict, method_list):
    for method in method_list:
        metric_dict[method] = 0.0

#############################

### set up ###
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
em = {}
es = {}
set_up_metric_dict(em, method_list)
set_up_metric_dict(es, method_list)
output_texts = {}
for method in method_list:
    output_texts[method] = []

## run ##
for step, (input_ids, gt) in tqdm(enumerate(test_dataloader)):
    input_ids = input_ids.to(device)
    max_len = len(input_ids[0])+100

    ### Greedy ###
    if 'greedy' in method_list:
        greedy_output = model.generate(input_ids, max_length=max_len)
        greedy_text = strip_to_line_completion_output(greedy_output[0], input_ids[0], tokenizer)
        output_texts['greedy'].append(greedy_text)
        es['greedy'] += fuzz.ratio(greedy_text, gt[0])
        em['greedy'] += 1 if greedy_text == gt[0] else 0

    ### Beam Search ###
    if 'beam' in method_list:
        beam_outputs = model.generate(
            input_ids, 
            max_length=max_len, 
            num_beams=beam_size, 
            no_repeat_ngram_size=no_repeat_ngram_size, 
            num_return_sequences=num_return_sequences, 
            early_stopping=early_stopping
        )
        beam_text = strip_to_line_completion_output(beam_outputs[0], input_ids[0], tokenizer)
        output_texts['beam'].append(beam_text)
        es['beam'] += fuzz.ratio(beam_text, gt[0])
        em['beam'] += 1 if beam_text == gt[0] else 0

    ### Sampling ###
    if 'sampling' in method_list:
        # activate sampling and deactivate top_k by setting top_k sampling to 0
        sample_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=max_len, 
            top_k=0
        )
        sampling_text = strip_to_line_completion_output(sample_output[0], input_ids[0], tokenizer)
        output_texts['sampling'].append(sampling_text)
        es['sampling'] += fuzz.ratio(sampling_text, gt[0])
        em['sampling'] += 1 if sampling_text == gt[0] else 0

    ### Sampling with temperature ###
    if 'sampling_temp' in method_list:
        # use temperature to decrease the sensitivity to low probability candidates
        sample_temp_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=max_len, 
            top_k=0, 
            temperature=temperature
        )
        sampling_temp_text = strip_to_line_completion_output(sample_temp_output[0], input_ids[0], tokenizer)
        output_texts['sampling_temp'].append(sampling_temp_text)
        es['sampling_temp'] += fuzz.ratio(sampling_temp_text, gt[0])
        em['sampling_temp'] += 1 if sampling_temp_text == gt[0] else 0

    ### Top-K Sampling ###
    if 'top_k' in method_list:
        top_k_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=max_len, 
            top_k=k_size
        )
        top_k_text = strip_to_line_completion_output(top_k_output[0], input_ids[0], tokenizer)
        output_texts['top_k'].append(top_k_text)
        es['top_k'] += fuzz.ratio(top_k_text, gt[0])
        em['top_k'] += 1 if top_k_text == gt[0] else 0

    ### Top-P (nucleus) Sampling ###
    if 'top_p' in method_list:
        # deactivate top_k sampling and sample only from 92% most likely words
        top_p_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=max_len, 
            top_p=p_size, 
            top_k=0
        )
        top_p_text = strip_to_line_completion_output(top_p_output[0], input_ids[0], tokenizer)
        output_texts['top_p'].append(top_p_text)
        es['top_p'] += fuzz.ratio(top_p_text, gt[0])
        em['top_p'] += 1 if top_p_text == gt[0] else 0

    # ### multiple independently sampled outputs ####
    # # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    # sample_outputs = model.generate(
    #     input_ids,
    #     do_sample=True, 
    #     max_length=max_len, 
    #     top_k=k_size, 
    #     top_p=p_size, 
    #     num_return_sequences=num_return_sequences
    # )

    if step % 100 == 0:
        print(f" {step} are done!")

    # break

for method in method_list:
    saved_text_file = output_dir + f'{method}_text_outputs.txt'
    with open(saved_text_file, "w") as f:
        preds = output_texts[method]
        for pred_text in preds:
            f.write(pred_text+"\n")
        f.write(f'save texts to file: {saved_text_file}')
    print(f'save texts to file: {saved_text_file}')

saved_results_file = output_dir + f'decoding_methods_results.txt'
with open(saved_results_file, "w") as f:
    len_data = len(dataset)
    f.write(f"Results from pretrain dir: {model_dir}\n")
    f.write(f"Test {len_data} samples\n")
    for method in method_list:
        em_result = em[method]
        es_result = es[method]
        f.write(f"{method} \t\t:: Edit sim: {es_result/len_data}, EM: {em_result/len_data}"+"\n")
    f.write(f"Config:\nmethods:{method_list}\nBeam_size={beam_size}, temperature={temperature}, k_size={k_size}, p_size={p_size},\nno_token_types={no_token_types}, no_repeat_ngram_size={no_repeat_ngram_size}, seed_number={seed_number}\n")
    f.write(f'save results to file: {saved_results_file}')
print(f'save results to file: {saved_results_file}')
print('Complete.')