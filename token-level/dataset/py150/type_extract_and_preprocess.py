# This code is modify from CodeXGLUE from Microsoft.
# Licensed under the MIT License.

import os
import argparse
import re
from tokenize import tokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER, DEDENT, ERRORTOKEN, NAME
import tokenize as tk
from io import BytesIO
import json
import keyword

lits = json.load(open("literals.json"))

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def preprocess_dataset(args, file_name, file_type):
    #### extract exact token type from tokenzier library ####
    transform_dict = {
        NL: "<EOL>",
        NEWLINE: "<EOL>",
        INDENT: "<INDENT>",
        DEDENT: "<DEDENT>",
    }
    file_paths = open(os.path.join(args.base_dir, file_name)).readlines()
    wf_code = open(os.path.join(args.output_dir, f"{file_type}.txt"), 'w')
    wf_type_exact = open(os.path.join(args.output_dir, f"type_{file_type}.txt"), 'w')
    n_error = 0
    for ct,path in enumerate(file_paths):
        out_code = []
        toktype_exact = []
        try:
            code = open(os.path.join(args.base_dir, path.strip())).read()
            token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
            was_eol = False
            for tok in token_gen:
                toknum = tok.type
                toknum_exact = tok.exact_type
                tokval = " ".join(tok.string.split())
                if toknum == ERRORTOKEN and tokval in [" ",""]:
                    continue
                elif toknum in [NEWLINE, NL]:
                    if not was_eol:
                        out_code.append("<EOL>")
                        toktype_exact.append("<EOL>")
                        was_eol = True
                elif toknum in transform_dict:
                    out_code.append(transform_dict[toknum])
                    toktype_exact.append("<" +tk.tok_name[toknum_exact]+ ">")
                    was_eol = False
                elif toknum == NAME and keyword.iskeyword(tokval):
                    out_code.append(tokval)
                    toktype_exact.append("<KEYWORD>")
                    was_eol = False
                elif toknum == STRING:
                    add_token = process_string(tokval)
                    out_code.append(add_token)
                    toktype_exact.append("<" +tk.tok_name[toknum_exact]+ ">")
                    was_eol = False
                elif toknum == NUMBER: 
                    if tokval in lits['num']:
                        out_code.append(f"<NUM_LIT:{tokval}>")
                    else:
                        out_code.append(f"<NUM_LIT>")
                    toktype_exact.append("<" +tk.tok_name[toknum_exact]+ ">")
                    was_eol = False
                elif toknum not in [COMMENT, ENCODING, ENDMARKER]:
                    out_code.append(tokval)
                    toktype_exact.append("<" +tk.tok_name[toknum_exact]+ ">")
                    was_eol = False
            if len(out_code) > 0 and out_code[0] == "<EOL>":
                out_code = out_code[1:]
                toktype_exact = toktype_exact[1:]
            # if len(out_code) > 0 and out_code[-1] == "<EOL>":
            #     out_code = out_code[:-1]
            #     toktype_exact = toktype_exact[:-1]
        except Exception as e:
            n_error += 1
            pass
        if len(out_code) > 0 and out_code[0] == "<EOL>":
            out_code.append("<EOL>")
            toktype_exact.append("<EOL>")
        # break
        out_code = ["<s>"] + out_code + ["</s>"]
        out = " ".join(out_code)
        wf_code.write(out+"\n")

        toktype_exact = ["<s>"] + toktype_exact + ["</s>"]
        exact = " ".join(toktype_exact)
        wf_type_exact.write(exact+"\n")
        if ct % 10000 == 0:
            print(f"{file_type}: {ct} are done")
    print(f"{n_error} are error")
    wf_code.close()
    wf_type_exact.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[:-5000]
    dev_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[-5000:]
    wf = open(os.path.join(args.base_dir, "python95k_train.txt"), "w")
    for path in train_paths:
        wf.write(path)
    wf.close()
    wf = open(os.path.join(args.base_dir, "python5k_dev.txt"), "w")
    for path in dev_paths:
        wf.write(path)
    wf.close()

    preprocess_dataset(args, file_name="python95k_train.txt", file_type="train")
    preprocess_dataset(args, file_name="python5k_dev.txt", file_type="dev")
    preprocess_dataset(args, file_name="python50k_eval.txt", file_type="test")

if __name__ == "__main__":
    main()
