import pickle
import torch
import numpy as np
import os
import argparse
import torch
import tqdm 
import pandas as pd

from style_paraphrase.inference_utils import GPT2Generator
from style_paraphrase.dataset_config import DATASET_CONFIG, BASE_CONFIG
from style_paraphrase.data_utils import update_config, Instance, get_label_dict 
from style_paraphrase.utils import init_gpt2_model 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
#parser.add_argument("--model_dir", default="/home/hyunji/style-transfer-paraphrase/style_paraphrase/saved_models/random_pos/checkpoint-3500/", type=str)
#parser.add_argument("--model_dir", default="/home/hyunji/style-transfer-paraphrase/style_paraphrase/saved_models/possstag/checkpoint-18500/", type=str)
parser.add_argument("--model_dir", default="/home/hyunji/style-transfer-paraphrase/style_paraphrase/saved_models/para_tone_paws_paraphrase_model/checkpoint-45183/", type=str)
#parser.add_argument("--model_dir", default="/home/hyunji/style-transfer-paraphrase/style_paraphrase/saved_models/sample_random_pos_tone/checkpoint-2700/", type=str)
#parser.add_argument("--model_dir", default="/home/hyunji/style-transfer-paraphrase/style_paraphrase/saved_models/random_pos_tone/checkpoint-500/", type=str)
parser.add_argument("--paraphrase_str", default="paraphrase_250", type=str)
parser.add_argument("--top_p_value", default=0.0, type=float)
args = parser.parse_args()

roberta = torch.hub.load("pytorch/fairseq", "roberta.base", force_reload=True)

if not torch.cuda.is_available():
   print("Please check if a GPU is available or your Pytorch installation is correct!")
   sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir, upper_length="same_5", is_korean=True)
paraphraser.modify_p(top_p=args.top_p_value)

"""
data_path = "./datasets/abstract/full.txt"
label_path = "./datasets/abstract/full.label"

with open(data_path, "r") as f:
   data = f.read().strip().split("\n")[:100]

assert len(data) == len(labels)
"""

paraphrased_csv = "/home/hyunji/style-transfer-paraphrase/examples_100.csv"
df = pd.read_csv(paraphrased_csv)
data = list(df['original'])

ret_dict = {'paraphrased': data, 'label_1': []}

outputs = []
for i in tqdm.tqdm(range(len(data))):
   generations, _ = paraphraser.generate_batch([data[i]])
   ret_dict['label_1'].append(generations[-1])
assert(len(ret_dict['label_1']) == len(data))

df = pd.DataFrame(ret_dict)
#df.to_csv("./random_pos.csv")
#df.to_csv("./possstag_ckpt_18500.csv")
#df.to_csv("./pos_and_para.csv")
#df.to_csv("./sample_pos_tone_2700.csv")
#df.to_csv("./random_pos_tone.csv")
df.to_csv("./para_tone_paws_paraphrase_model.csv")
