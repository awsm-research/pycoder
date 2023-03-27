# PyCoder

This repository contains the code for the paper [Syntax-Aware On-the-Fly Code Completion](https://arxiv.org/abs/2211.04673)

PyCoder leverage a Multi-Task Training technique (MTT) to cooperatively
learn the code prediction task and the type prediction task. For the type prediction
task, we propose to leverage the standard Python token
type information (e.g., String, Number, Name, Keyword),
which is readily available and lightweight, instead of using
the AST information which requires source code to be parsable for an extraction, limiting its ability to perform on-the-fly code completion (see Section 2.3 in our paper). 

The overview of our PyCoder is shown in the figure below. More information can be found in our paper.
![An overview of our Syntax-Aware On-the-Fly Python Code Completion approach (PyCoder).](https://github.com/awsm-research/pycoder/blob/main/assets/images/overview.png)

The results of our PyCoder is in `assets/notebooks/results.ipynb`

## Setting Up

### Dependency
- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0 and < 4.0.0
- fuzzywuzzy

## Dataset

We use PY150 in our experiments and follow the data splitting from CodeXGLUE. Data statistics of PY150 dataset are shown in the below tables

* For token-level prediction:

| Data Split  |   #Files    |   #Tokens   |
| ----------- | :---------: | :---------: |
|    Train    |    95,000   |    72.1M    |
|     Dev     |    5,000    |     4.4M    |
|    Test     |    50,000   |    37.3M    |

* For line-level prediction:

| Data Split |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    10,000   |          477.81          |          6.61             |

Our already processed dataset (code + token type dataset) is available in HuggingFace: [PyCoder Dataset](https://huggingface.co/datasets/Wannita/PyCoder/tree/main).

To download and preprocess the dataset by yourself, navigate to `token-level/dataset/py150` directory, and run
```shell
bash download_and_extract.sh
python type_extract_and_preprocess.py --base_dir=py150_files --output_dir=token_completion
python type_alignment.py --base_dir=token_completion --output=token_completion
```

## Training

We build three variants of PyCoder, with
three different MTT techniques, according to two learning styles. Respectively, our best model is PyCoder-Hard, PyCoder-IFN, and PyCoder-Soft.

To fine-tune (the example is PyCoder-Hard), navigate to `token-level/code` directory, run:

```
LANG=python          
DATADIR=../dataset/py150/token_completion
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=../save/<output_dir> # model saved here
PRETRAINDIR=microsoft/CodeGPT-small-py
LOGFILE=<log_dir>.log # log file saved here
PER_NODE_GPU=1

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_pycoder.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=15 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain \
        --train_mode=both \
        --model_amount=1 \
        --loss_weight_value=0.9
```

The arguments can be modify as follow:
* **PyCoder-Hard**:
```
 --train_mode=both \
 --model_amount=1 \
 --loss_weight_value=0.9 #change weight for code prediction task, type prediction will be (1-weight)
```

* **PyCoder-Soft**:
```
 --train_mode=both \ 
 --model_amount=2 \
 --soft_share \
 --loss_weight_mode=none 
```

* **PyCoder-IFN**:
```
 --train_mode=type \ #change to 'code' later
 --model_amount=1 
```

## Evaluation

### Token-level

```
LANG=python                      
DATADIR=../dataset/py150/token_completion
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=../save/<output_dir> # predictions saved here
PRETRAINDIR=../save/<model_dir>/<checkpoint_file> #  directory of your saved model
LOGFILE=<log_dir>.log # log file saved here

python -u run_pycoder.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=4 \
        --logging_steps=100 \
        --seed=42 \
        --predict_mode=code 
```

### Line-level

```
LANG=python                      
DATADIR=../dataset/py150/line_completion
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=../../line-level/save/<output_dir> # predictions saved here
PRETRAINDIR=../../line-level/save/<model_dir>/<checkpoint_file> # directory of your saved model
LOGFILE=<log_dir>.log # log file saved here

python -u run_pycoder_line.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --predict_mode=code \
        --calculate_mrr
```

## Inference only

Our trained model is available in HuggingFace: [PyCoder](https://huggingface.co/Wannita/PyCoder).

To find the sample usage code, navigate to `assets/notebooks/inference.ipynb` directory.

## Reference

If you use our code or PyCoder, please cite our [PyCoder paper](https://arxiv.org/abs/2211.04673).

For PyCoder:

<pre><code>@article{takerngsaksiri2022syntax,
  title={Syntax-Aware On-the-Fly Code Completion},
  author={Takerngsaksiri, Wannita and Tantithamthavorn, Chakkrit and Li, Yuan-Fang},
  journal={arXiv preprint arXiv:2211.04673},
  year={2022}
}</code></pre>


Additionally, please also cite the following papers in addition to our PyCoder.

For CodeXGLUE:

<pre><code>@article{DBLP:journals/corr/abs-2102-04664,
  author    = {Shuai Lu and Daya Guo and Shuo Ren and Junjie Huang and Alexey Svyatkovskiy and Ambrosio Blanco and Colin B. Clement and Dawn Drain and Daxin Jiang and Duyu Tang and Ge Li and Lidong Zhou and Linjun Shou and Long Zhou and Michele Tufano and Ming Gong and Ming Zhou and Nan Duan and Neel Sundaresan and Shao Kun Deng and Shengyu Fu and Shujie Liu},
  title     = {CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding
               and Generation},
  journal   = {CoRR},
  volume    = {abs/2102.04664},
  year      = {2021}
}</code></pre>

For PY150 dataset:

<pre><code>@article{raychev2016probabilistic,
  title={Probabilistic Model for Code with Decision Trees},
  author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
  journal={ACM SIGPLAN Notices},
  pages={731--747},
  year={2016},
  publisher={ACM New York, NY, USA}
}</code></pre>

