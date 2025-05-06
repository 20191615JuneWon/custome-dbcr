#!/usr/bin/env python
import os
import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.utils as vutils
from PIL import Image

from ddbm import dist_util, logger
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ddbm.karras_diffusion import karras_sample
from datasets import load_data

def main(args):
    dist_util.setup_dist()
    device = dist_util.dev()
    workdir = Path(args.model_path).parent
    workdir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(workdir))

    step = int(Path(args.model_path).stem.rsplit("_",1)[-1])
    sample_dir = workdir / f"sample_{step}_w{args.guidance}_churn{args.churn_step_ratio}"
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    md_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**md_kwargs)
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(device)
    if args.use_fp16:
        model.half()
    model.eval()

    train_loader, val_loader, test_loader = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        include_test=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.split == "train":
        dataloader = train_loader
    elif args.split == "test":
        dataloader = test_loader
    else:
        raise ValueError("split must be 'train' or 'test'")

    num_samples = len(dataloader.dataset)
    all_images = []
    nfe = None

    for batch_idx, batch in enumerate(dataloader):

        x0_image, y0_image = batch
        x0 = (x0_image.to(device) * 2 - 1).to(device)
        y0 = (y0_image.to(device) * 2 - 1).to(device)

        x_out, path, nfe = karras_sample(
            diffusion    = diffusion,
            model        = model,
            x_T          = y0,
            x_0          = x0,
            steps        = args.steps,
            clip_denoised= args.clip_denoised,
            progress     = False,
            callback     = None,
            model_kwargs = None,  # conditional 키가 필요하면 수정
            device       = device,
            sigma_min    = diffusion.sigma_min,
            sigma_max    = diffusion.sigma_max,
            rho          = args.rho,
            sampler      = args.sampler,
            churn_step_ratio = args.churn_step_ratio,
            guidance     = args.guidance,
        )

        sample_uint8 = ((x_out + 1) * 127.5).clamp(0,255).to(th.uint8)
        sample_np = sample_uint8.permute(0,2,3,1).cpu()

        gathered = [th.zeros_like(sample_uint8) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, sample_uint8)
        gathered = th.cat(gathered, dim=0)
        all_images.append(gathered.cpu().numpy())

        if batch_idx == 0 and dist.get_rank() == 0:
            B = sample_uint8.shape[0]
            nrow = int(np.sqrt(min(32, B)))
            vutils.save_image(
                (sample_uint8[:32].float() / 255.0).permute(0,3,1,2),
                str(sample_dir / f"sample_{batch_idx}.png"),
                nrow=nrow,
            )
            vutils.save_image(
                x0_image[:32],
                str(sample_dir / f"x0_{batch_idx}.png"),
                nrow=nrow,
            )
            vutils.save_image(
                (y0_image[:32] / 2 + 0.5),
                str(sample_dir / f"y0_{batch_idx}.png"),
                nrow=nrow,
            )

    all_arr = np.concatenate(all_images, axis=0)[:num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join(map(str, all_arr.shape))
        out_path = sample_dir / f"samples_{shape_str}_nfe{nfe}.npz"
        logger.log(f"Saving to {out_path}")
        np.savez(out_path, all_arr)

    dist.barrier()
    if dist.get_rank() == 0:
        logger.log("Sampling complete")

    if dist.is_initialized():
        dist.destroy_process_group()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset="edges2handbags",
        clip_denoised=True,
        batch_size=16,
        sampler="heun",
        split="train",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=40,
        model_path="",
        seed=42,
        num_workers=2,
        guidance=1.0,
        use_fp16=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
