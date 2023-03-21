from transformers import GPT2Tokenizer
import json
from tqdm import tqdm
import argparse
import os

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

def create_repeat_types(tokens_list, types_lists, tokenizer):
    new_gts_type = []
    for j in tqdm(range(len(tokens_list))):
        gt_types = types_lists[j].split(' ')
        gt_tokens = tokenizer.tokenize(tokens_list[j].strip())
        new_gt_types = []
        i = 0
        next_type = False
        for tok in gt_tokens:
            if tok[0] == 'Ġ' or tok.startswith('<NUM_LIT') or next_type:
                i+=1
                next_type = False
                if tok in ['<EOL>','<INDENT>','<DEDENT>'] or tok.startswith('<NUM_LIT'):
                    next_type = True
            elif tok in ['<EOL>','<INDENT>','<DEDENT>'] or tok.startswith('<NUM_LIT'):
                next_type = True
                i+=1
            new_gt_types.append(gt_types[i])
        if new_gt_types[-2].strip() == '</s>':
            new_gt_types = new_gt_types[:-1]
        if gt_types[i].strip() != '</s>' or tok.strip() != '</s>':
            print(j, 'Error')
            break
        new_gts_type.append(' '.join(new_gt_types))
    return new_gts_type

def save_file_txt(file_name, data_list, newline=True):
    with open(file_name, 'w') as f:
        for line in data_list:
            line = line + '\n' if newline else line
            f.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="token_completion", type=str, 
                        help="The data directory to code and type")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    gts_token_test = open(os.path.join(args.base_dir, "test.txt")).readlines()
    gts_type_test = open(os.path.join(args.base_dir, "type_test.txt")).readlines() 
    gts_token_train = open(os.path.join(args.base_dir, "train.txt")).readlines()
    gts_type_train = open(os.path.join(args.base_dir, "type_train.txt")).readlines()
    gts_token_dev = open(os.path.join(args.base_dir, "dev.txt")).readlines()
    gts_type_dev = open(os.path.join(args.base_dir, "type_dev.txt")).readlines()

    special_tokens = get_special_tokens('literals.json')
    special_tokens.extend(['<NAME>', '<KEYWORD>', '<NUMBER>', '<STRING>', '<NEWLINE>', '<INDENT>', '<DEDENT>', '<LPAR>', '<RPAR>', '<LSQB>', '<RSQB>', '<COLON>', '<COMMA>', '<SEMI>', '<PLUS>', '<MINUS>', '<STAR>', '<SLASH>', '<VBAR>', '<AMPER>', '<LESS>', '<GREATER>', '<EQUAL>', '<DOT>', '<PERCENT>', '<LBRACE>', '<RBRACE>', '<EQEQUAL>', '<NOTEQUAL>', '<LESSEQUAL>', '<GREATEREQUAL>', '<TILDE>', '<CIRCUMFLEX>', '<LEFTSHIFT>', '<RIGHTSHIFT>', '<DOUBLESTAR>', '<PLUSEQUAL>', '<MINEQUAL>', '<STAREQUAL>', '<SLASHEQUAL>', '<PERCENTEQUAL>', '<AMPEREQUAL>', '<VBAREQUAL>', '<CIRCUMFLEXEQUAL>', '<LEFTSHIFTEQUAL>', '<RIGHTSHIFTEQUAL>', '<DOUBLESTAREQUAL>', '<DOUBLESLASH>', '<DOUBLESLASHEQUAL>', '<AT>', '<ATEQUAL>', '<RARROW>', '<ELLIPSIS>', '<ERRORTOKEN>'])
    tokenizer = GPT2Tokenizer.from_pretrained('microsoft/CodeGPT-small-py', do_lower_case=False, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)

    repeat_gts_type_dev = create_repeat_types(gts_token_dev, gts_type_dev, tokenizer)
    repeat_gts_type_test = create_repeat_types(gts_token_test, gts_type_test, tokenizer)
    repeat_gts_type_train = create_repeat_types(gts_token_train, gts_type_train, tokenizer)

    save_file_txt(os.path.join(args.output_dir, "type_repeat_dev.txt"), repeat_gts_type_dev, newline=False)
    save_file_txt(os.path.join(args.output_dir, "type_repeat_test.txt"), repeat_gts_type_test, newline=False)
    save_file_txt(os.path.join(args.output_dir, "type_repeat_train.txt"), repeat_gts_type_train, newline=False)

if __name__ == "__main__":
    main()