import nnsight
from nnsight import LanguageModel
# from transformer_lens import HookedTransformer
import torch
import transformers
import json
from tqdm import tqdm
import yaml


from transformer_lens.evals import make_wiki_data_loader
from transformer_lens.HookedTransformer import HookedTransformer
import logging
import os
import pickle as pkl
from argparse import ArgumentParser