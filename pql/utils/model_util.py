import wandb
import torch
from loguru import logger
from pathlib import Path
import pql
from pql.utils.common import load_class_from_path


def load_model(model, model_type, cfg):
    artifact = wandb.Api().artifact(cfg.artifact)
    artifact.download(pql.LIB_PATH)
    logger.warning(f'Load {model_type}')
    weights = torch.load(Path(pql.LIB_PATH, "model.pth"))

    try:
        if model_type == "obs_rms" and weights[model_type] is None:
            logger.warning(f'Observation normalization is enabled, but loaded weight contains no normalization info.')
            return
        model.load_state_dict(weights[model_type])
    except KeyError:
        logger.warning(f'Invalid model type:{model_type}')


def save_model(path, actor, critic, rms, wandb_run, description):
    if isinstance(actor, list):
        checkpoint = {'obs_rms': rms,
            'critic': critic
            }
        for i in range(len(actor)):
            checkpoint[f'actor_{i}'] = actor[i].state_dict()
    else:
        checkpoint = {'obs_rms': rms,
            'actor': actor,
            'critic': critic
            }
    torch.save(checkpoint, path)  # save policy network in *.pth

    model_artifact = wandb.Artifact(wandb_run.id, type="model", description=description)
    model_artifact.add_file(path)
    wandb.save(path, base_path=wandb_run.dir)
    wandb_run.log_artifact(model_artifact)