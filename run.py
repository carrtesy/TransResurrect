import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import wandb

from utils.tools import SEED_everything
from utils.argpass import prepare_arguments

def main():
    SEED_everything(2023)
    parser = argparse.ArgumentParser(description='TransResurrect')
    args = prepare_arguments(parser)

    wandb.login()
    WANDB_PROJECT_NAME, WANDB_ENTITY = "TransResurrect", "carrtesy"

    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
    wandb.config.update(args)

    Exp_Main(args).train()
    Exp_Main(args).test()

if __name__ == "__main__":
    main()
