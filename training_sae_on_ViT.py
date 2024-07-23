import os
import sys
import torch
import wandb
import json
import pickle
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path
from tqdm import tqdm
from functools import partial


sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from vit_sae_analysis.dashboard_fns import get_feature_data

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

cfg = ViTSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    class_token = True,
    image_width = 224,
    image_height = 224,
    model_name = "openai/clip-vit-large-patch14",
    module_name = "resid",
    block_layer = -2,
    dataset_path = "evanarlian/imagenet_1k_resized_256",
    use_cached_activations = False,
    cached_activations_path = None,
    d_in = 1024,

    # SAE Parameters
    expansion_factor = 32,
    b_dec_init_method = "geometric_median",
    # gated_sae = True,
    gated_sae=False,

    # Training Parameters
    lr = 0.0004,
    # l1_coefficient = 0.0,
    l1_coefficient = 0.00008,
    # l1_coefficient = 0.0008,
    lr_scheduler_name="constantwithwarmup",
    batch_size = 1024,
    lr_warm_up_steps=500,
    total_training_tokens = 2_621_440,
    # total_training_tokens = 1_536_000,
    # total_training_tokens = 10240,
    n_batches_in_store = 15,

    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 64,
    dead_feature_window=64,
    dead_feature_threshold = 1e-6,

    # WANDB
    log_to_wandb = True,
    wandb_project= "mats-hugo",
    wandb_entity = None,
    wandb_log_frequency=20,

    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 0,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )

torch.cuda.empty_cache()
### resume
resume_cfg = {
    "run_name" : "yzdd34k7",
    "start_training_steps" : 1750,
    "pt_name" : "clip-vit-large-patch14_-2_resid_32768.pt",
}
ckpt_path = os.path.join("checkpoints", resume_cfg['run_name'], f"{resume_cfg['start_training_steps']}_sparse_autoencoder_openai", resume_cfg["pt_name"])
resume_cfg['ckpt_path'] = ckpt_path
cfg = torch.load(ckpt_path)['cfg']
sparse_autoencoder, model = vision_transformer_sae_runner(cfg, resume=True, resume_cfg=resume_cfg)
### from scratch
# sparse_autoencoder, model = vision_transformer_sae_runner(cfg, resume=False)
sparse_autoencoder.eval()


get_feature_data(
    sparse_autoencoder,
    model,
    number_of_images = 524_288,
    number_of_max_activating_images = 20,
)

print("*****Done*****")
