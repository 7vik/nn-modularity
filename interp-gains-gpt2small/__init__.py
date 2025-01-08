import nnsight
from nnsight import LanguageModel
# from transformer_lens import HookedTransformer
import torch
import transformers
from transformers import GPT2Model, GPT2Tokenizer
import json
from tqdm import tqdm
import yaml
from matplotlib import pyplot as plt
import seaborn as sns


from transformer_lens.evals import make_wiki_data_loader
from transformer_lens.HookedTransformer import HookedTransformer
import logging
import os
import pickle as pkl
from argparse import ArgumentParser