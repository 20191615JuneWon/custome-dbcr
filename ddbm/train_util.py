import copy
import functools
import os

from pathlib import Path

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from torchvision.utils import save_image

from . import dist_util, logger
from .nn import update_ema

from ddbm.random_util import get_generator
import numpy as np
from ddbm.script_util import NUM_CLASSES

from ddbm.karras_diffusion import karras_sample

import glob 



class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        sample_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.workdir = workdir
        self.sample_interval = sample_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=fp16_scale_growth,
        # )

        self.scaler = th.amp.GradScaler('cuda', enabled=self.use_fp16)

        self.dtype = th.float16 if use_fp16 else th.float32
        self.device = dist_util.dev()

        self.opt = RAdam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                find_unused_parameters=True
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs['generator'], self.batch_size, sample_kwargs['seed'])
        self.sample_kwargs = sample_kwargs
        self.augment = augment_pipe
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log('Resume step: ', self.resume_step)
                
            self.model.load_state_dict(
                # dist_util.load_state_dict(
                #     resume_checkpoint, map_location=dist_util.dev()
                # ),
                th.load(resume_checkpoint, map_location=dist_util.dev()),
            )
        
            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # state_dict = dist_util.load_state_dict(
            #     ema_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        base = os.path.basename(main_checkpoint)
        if base.startswith('latest_'):
            prefix = "latest_"
        elif base.startswith('freq_'):
            prefix = 'freq_'
        else:
            prefix = ''

        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), 
            f"{prefix}opt_{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()

    def preprocess(self, x):
        x =  x * 2 - 1                
        return x

    def run_loop(self):
        while True:
            for (opt, sar), target in self.data:
                # TODO:
                if self.step >= self.total_training_steps:
                    (test_opt, test_sar), test_target = next(iter(self.test_data))
                    test_target = test_target.to(device=self.device, dtype=self.dtype)
                    test_opt    = self.preprocess(test_opt).to(device=self.device, dtype=self.dtype)
                    test_sar    = self.preprocess(test_sar).to(device=self.device, dtype=self.dtype)
                    
                    test_cond = {'opt':test_opt, 'sar':test_sar}

                    self.sample_and_save(test_cond, test_target)
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                target = target.to(device=self.device, dtype=self.dtype)
                opt    = self.preprocess(opt).to(device=self.device, dtype=self.dtype)
                sar    = self.preprocess(sar).to(device=self.device, dtype=self.dtype)

                batch = target
                cond = {'opt': opt, 'sar': sar}
                    
                took_step = self.run_step(batch, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()     

                if self.step % self.sample_interval == 0:
                    (test_opt, test_sar), test_target = next(iter(self.test_data))

                    test_target = test_target.to(device=self.device, dtype=self.dtype)
                    test_opt    = self.preprocess(test_opt).to(device=self.device, dtype=self.dtype)
                    test_sar    = self.preprocess(test_sar).to(device=self.device, dtype=self.dtype)

                    test_batch = test_target
                    test_cond = {'opt':test_opt, 'sar':test_sar}
                    self.run_test_step(test_batch, test_cond)

                    logger.dumpkvs()

                    self.sample_and_save(test_cond, test_batch)

                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)
        

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        logger.logkv_mean("lg_loss_scale", np.log2(self.scaler.get_scale()))
        self.scaler.unscale_(self.opt)

        def _compute_norms():
            grad_norm = 0.0
            param_norm = 0.0
            for p in self.model.parameters():
                with th.no_grad():
                    param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                    if p.grad is not None:
                        grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
            return np.sqrt(grad_norm), np.sqrt(param_norm)
        
        grad_norm, param_norm = _compute_norms()

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        self._update_ema()

        self._anneal_lr()
        self.log_step()
        return True

    def run_test_step(self, batch, cond):
        with th.no_grad():
            self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.opt.zero_grad()

        x0    = batch.to(device=self.device, dtype=self.dtype)
        opt   = cond['opt'].to(device=self.device, dtype=self.dtype)
        sar   = cond['sar'].to(device=self.device, dtype=self.dtype)

        sub_cond = {'opt': opt, 'sar': sar, 'x0': x0}
        num_microbatches = batch.shape[0] // self.microbatch
        for i in range(0, batch.shape[0], self.microbatch):
            with th.autocast(device_type="cuda", dtype=th.float16, enabled=self.use_fp16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
                last_batch = (i + self.microbatch) >= batch.shape[0]

                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                print("train_util, t", t.min().cpu().item(), t.min().cpu().item())
                compute_losses = functools.partial(
                    self.diffusion.training_bridge_losses,
                    self.ddp_model,
                    t,
                    model_kwargs=sub_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                loss = (losses["loss"] * weights).mean() / num_microbatches
            log_loss_dict(self.diffusion, t, {k if train else "test_" + k: v * weights for k, v in losses.items()})
            if train:
                self.scaler.scale(loss).backward()



    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr


    def _log_losses(self, t, losses, prefix=''):
        logger.logkv(f"{prefix}step", self.step)

        try:
            sig_mean = float(t.mean().item())
            logger.logkv(f"{prefix}sigma_mean", sig_mean)
        except Exception:
            pass

        for name, val in losses.items():
            if val.dim() > 0:
                v = val.mean().item()
            else:
                v = val.item()
            logger.logkv(f"{prefix}{name}", float(v))
                
    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f'{(self.step):06d}')[0]+'*'
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3:
                earliest = min(freq_states, key=lambda x: x.split('_')[-1].split('.')[0])
                os.remove(earliest)
                    
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)
                
            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()
    

    @th.no_grad()
    def sample_and_save(self, cond, target):
        device = self.device
        dtype  = self.dtype

        opt = cond['opt'].to(device=device, dtype=dtype)
        sar = cond['sar'].to(device=device, dtype=dtype)

        if target is None:
            raise ValueError("`target` (MS ground truth) must be provided for sampling.")
        
        ms_shape = target.shape
        x_t = th.randn(ms_shape, device=device, dtype=dtype) * self.diffusion.sigma_max

        opt_noise = opt + x_t
        sample, path, nfe = karras_sample(
            diffusion    = self.diffusion,
            model        = self.model,
            x_t          = opt_noise,
            x_0          = target.to(device, dtype),  
            steps        = self.step,
            clip_denoised= True,
            progress     = False,
            model_kwargs = {'opt': opt, 'sar': sar},
            device       = device,
        )

        sample = (sample + 1) * 0.5
        sample = sample.clamp(0, 1).cpu()

        out_dir = os.path.join(get_blob_logdir(), "samples")
        os.makedirs(out_dir, exist_ok=True)

        B, _, _, _ = sample.shape

        for i in range(B):
            img_ms = sample[i]
            rgb = img_ms[[3, 2, 1], :, :]
            fn = os.path.join(out_dir, f"sample_ms_{i}.png")
            save_image(rgb, fn, normalize=True)

        opt_vis = (opt + 1) * 0.5
        opt_vis = opt_vis.clamp(0, 1).cpu()
        for i in range(B):
            img_opt = opt_vis[i]
            img_opt = img_opt[[3, 2, 1], :, :]
            fn = os.path.join(out_dir, f"input_opt_{i}.png")
            save_image(img_opt, fn, normalize=True)

        # sar_vis = sar.clamp(0, 1).cpu()
        # for i in range(sar_vis.shape[0]):
        #     for b in range(sar_vis.shape[1]):
        #         img_sar = sar_vis[i, b : b+1, :, :]
        #         fn = os.path.join(out_dir, f"input_sar_{i}_band{b}.png")
        #         save_image(img_sar, fn, normalize=True)

        gt = target
        gt = gt.clamp(0, 1).cpu()
        for i in range(B):
            img_gt = gt[i]
            rgb = img_gt[[3, 2, 1], :, :]
            fn = os.path.join(out_dir, f"gt_{i}.png")
            save_image(rgb, fn, normalize=True)

        print(f"[inference] nfe={nfe}, saved {B} samples to {out_dir}")




def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.

    """
    base = os.path.basename(filename)      
    name = base.rsplit(".", 1)[0]
    if "_" not in name:
        return 0
    step_str = name.rsplit("_", 1)[1]
    try:
        return int(step_str)
    except ValueError:
        return 0

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split('/')[-1].startswith("latest"):
        prefix = 'latest_'
    elif main_checkpoint.split('/')[-1].startswith("freq"):
        prefix = 'freq_'
    else:
        prefix = ''
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

