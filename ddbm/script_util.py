# script_util.py

import argparse
import numpy as np

from .karras_diffusion import KarrasDenoiser
from .nafunet import NAFUNetModel

NUM_CLASSES = 1000

def get_workdir(exp):
    return f'./workdir/{exp}'

def model_and_diffusion_defaults():

    return dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0.,

        image_size=256,
        in_channels=13,                   # Sentinel-2 13개 밴드
        model_channels=22,                # base 채널 차원

        channel_mult="1,2,4,8",           # 각 레벨별 배수
        num_naf_blocks_enc="1,1,1,28",    # 인코더 레벨별 NAFBlock 수
        num_naf_blocks_dec="1,1,1,1",     # 디코더 레벨별 NAFBlock 수
        num_heads_per_level="1,1,2,4",    # SFBlock 헤드 수

        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,

        pred_mode="vp",                   # VE bridge
        weight_schedule="karras",
        rho=7.0,                          # Karras rho
    )

def sample_defaults():
    return dict(
        generator="determ",
        clip_denoised=True,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.002,
        s_tmax=80.0,
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        t = type(v)
        if v is None:
            t = str
        elif isinstance(v, bool):
            t = str2bool
        parser.add_argument(f"--{k}", default=v, type=t)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")

def create_model_and_diffusion(
    *,
    image_size,
    in_channels,
    model_channels=22,
    channel_mult=(1,2,4,8),
    num_naf_blocks_enc=(1,1,1,28),
    num_naf_blocks_dec=(1,1,1,1),
    num_heads_per_level=(1,1,2,4),

    dropout=0.0,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    use_fp16=False,

    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    cov_xy=0.,
    pred_mode="ve",
    weight_schedule="karras",
    rho=7.0,
):
    cm       = tuple(int(x) for x in channel_mult.split(","))
    # enc_blks = tuple(int(x) for x in num_naf_blocks_enc.split(","))
    # dec_blks = tuple(int(x) for x in num_naf_blocks_dec.split(","))
    heads    = tuple(int(x) for x in num_heads_per_level.split(","))

    model = NAFUNetModel(
        in_channels=in_channels,
        out_channels=(in_channels * 2 if False else in_channels),  # learn_sigma=False 고정
        model_channels=model_channels,
        channel_mult=cm,
        num_naf_blocks=1,

        num_heads_per_level=heads,
        dropout=dropout,
        dims=2,
        use_checkpoint=use_checkpoint,
        # use_scale_shift_norm=use_scale_shift_norm,
        use_fp16=use_fp16,
    )

    diffusion = KarrasDenoiser(
        sigma_data=sigma_data,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta_d=beta_d,
        beta_min=beta_min,
        cov_xy=cov_xy,
        image_size=image_size,
        weight_schedule=weight_schedule,
        pred_mode=pred_mode,
        rho=rho,
    )

    return model, diffusion
