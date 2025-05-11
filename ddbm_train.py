import os
import argparse
from pathlib import Path

import torch as th
import torch.distributed as dist
from torchinfo import summary

import wandb

from ddbm import dist_util, logger
from datasets import load_data

from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir,
)
from ddbm.train_util import TrainLoop
from datasets.augment import AugmentPipe

def main(args):
    workdir = get_workdir(args.exp)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    device = dist_util.dev()

    logger.configure(dir=workdir)

    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + "_resume"
        wandb.init(project="dbcr", group=args.exp, name=name, config=vars(args), mode='online' if not args.debug else 'disabled')
        logger.log("creating model and diffusion...")

    md_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**md_kwargs)

    model.to(device)
    if args.use_fp16:
        model.half()

    if hasattr(diffusion, "to"):
        diffusion.to(device)

    if dist.get_rank() == 0:
        wandb.watch(model, log='all')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.batch_size == -1:
        per_gpu = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0 and dist.get_rank() == 0:
            logger.log(
                f"Warning: global_batch_size {args.global_batch_size} not divisible by world_size {dist.get_world_size()}, using {per_gpu*dist.get_world_size()} instead."
            )
    else:
        batch_size = args.batch_size

    if dist.get_rank() == 0:
        logger.log("Creating data loader...")

    data, test_data = load_data(
        frac=args.frac,
        seed=args.seed,
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    augment = None
    if args.use_augment:
        augment = AugmentPipe(
            p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
        )
    else:
        augment = None

    # 6) 학습 루프 실행
    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_training_steps=args.total_training_steps,
        augment_pipe=augment,
        **sample_defaults(),
    ).run_loop()
    if dist.get_rank() == 0:
        logger.log("Training complete.")

def create_argparser():
    defaults = dict(
        data_dir="/home/work/dataset/SEN12MSCR",
        dataset="sen12mscr",
        schedule_sampler="uniform",
        frac=0.1,
        seed=42,
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=285125, # 10000000
        global_batch_size=1,
        batch_size=4,                 # 40 -> 4 -> 8 x -> B * 23(image) * 256 * 256 -> 128 * 128
        microbatch=1,
        ema_rate="0.9999",
        log_interval=125,
        sample_interval=11405,
        save_interval=11405,
        save_interval_for_preemption=57025,
        resume_checkpoint="",
        exp="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=0,
        use_augment=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)