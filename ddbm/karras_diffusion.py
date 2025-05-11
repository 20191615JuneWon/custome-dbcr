"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS


from .nn import mean_flat, append_dims, append_zero

from functools import partial


def vp_logsnr(t, beta_d, beta_min, eps=1e-8):
    t = th.as_tensor(t)
    val = (0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1
    val = val.clamp(min=eps)
    return - th.log(val + eps)
    
def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

class KarrasDenoiser(nn.Module):
    """
    Diffusion bridge denoiser for VE/VP preds with Karras schedule.
    """
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
        cov_xy: float = 0.0,
        rho: float = 7.0,
        pred_mode: str = 'vp',        # 've', 'vp', 've_simple', ...
        weight_schedule: str = 'karras',
        loss_norm: str = 'lpips',
        num_timesteps: int = 40,
        image_size: int = 256,
        dtype=th.float32,
    ):
        super().__init__()
        self.sigma_data    = sigma_data
        self.sigma_max     = sigma_max
        self.sigma_min     = sigma_min
        self.beta_d        = beta_d
        self.beta_min      = beta_min
        self.cov_xy        = cov_xy
        self.rho           = rho
        self.pred_mode     = pred_mode
        self.weight_schedule = weight_schedule
        self.num_timesteps = 40
        self.image_size    = image_size

        self.sigma_data_end = self.sigma_data
        self.c = 1
        
        # loss
        self.loss_norm = loss_norm
        if loss_norm == 'lpips':
            self.lpips = LPIPS(replace_pooling=True, reduction='none')
        self.dtype = dtype

    # def to(self, dtype):
    #     self.dtype = dtype
    #     if self.loss_norm == 'lpips':
    #         self.lpips = self.lpips.to(dtype)
    #     return self

    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2
    
    def get_sigmas(self, sigmas):
        return sigmas

    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)

        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
                weightings = A / ((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )
            
            
            elif self.pred_mode == 'vp':
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                delta = (logsnr_T - logsnr_t + logs_t - logs_T)
                a_t = delta.exp().clamp(min=1e-8, max=1e8)
                expm1_val = th.expm1(logsnr_T - logsnr_t).clamp(min=1e-8, max=1e8)
                b_t = -expm1_val * logs_t.exp().clamp(min=1e-8, max=1e8)
                c_t = -expm1_val * (2*logs_t - logsnr_t).exp().clamp(min=1e-8, max=1e8)

                A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
                print("A", A)
                denominator = a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t
                weightings = A / (denominator + 1e-8)

            elif self.pred_mode == 'vp_sample' or self.pred_mode == 've_simple':

                weightings = th.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings


    def get_bridge_scalings(self, sigma):
        sigma_data = self.sigma_data
        sigma_data_end = self.sigma_data_end
        cov_xy = self.cov_xy
        sigma_max = self.sigma_max

        t = sigma
        alpha_t = th.ones_like(t)
        alpha_bar_t = alpha_t
        rho_t = t
        rho_T = sigma_max
        rho_bar_t = (rho_T**2 - rho_t**2).clamp(min=0).sqrt()
        a_t = (alpha_bar_t * rho_t**2) / rho_T**2
        b_t = (alpha_t * rho_bar_t**2) / rho_T**2
        c_t = (alpha_t * rho_bar_t * rho_t) / rho_T

        A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2 * a_t * b_t * cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * sigma_data**2 + a_t * cov_xy) / A
        c_out = (
            a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 * c_t**2
        ).clamp(min=1e-8).sqrt() * c_in

        c_in = c_in.clamp(min=1e-3, max=10)
        c_skip = c_skip.clamp(min=-1, max=1)
        c_out = c_out.clamp(min=1e-3, max=10)

        return c_skip, c_out, c_in


    def training_bridge_losses(
            self,
            model,
            sigmas,
            model_kwargs,
            noise=None
        ):
        assert model_kwargs is not None

        x0  = model_kwargs['x0'].to(self.dtype)
        opt = model_kwargs['opt'].to(self.dtype)
        sar = model_kwargs['sar'].to(self.dtype)

        if noise is None:
            noise = th.randn_like(x0)

        sigmas = th.minimum(sigmas, th.ones_like(sigmas)* self.sigma_max)
        sigmas = sigmas.clamp(min=self.sigma_min, max=self.sigma_max)

        xT = opt + self.sigma_max * noise
        terms = {}

        t = sigmas
        sigma_max = self.sigma_max
        alpha_t = th.ones_like(t)
        alpha_bar_t = alpha_t  # alpha_T = 1
        rho_t = t
        rho_T = sigma_max
        rho_bar_t = (rho_T**2 - rho_t**2).clamp(min=0).sqrt()

        a_t = (alpha_bar_t * rho_t**2) / rho_T**2
        b_t = (alpha_t * rho_bar_t**2) / rho_T**2
        c_t = (alpha_t * rho_bar_t * rho_t) / rho_T

        a_t = a_t.view(-1, *[1]*(x0.ndim-1))
        b_t = b_t.view(-1, *[1]*(x0.ndim-1))
        c_t = c_t.view(-1, *[1]*(x0.ndim-1))

        x_t = a_t * xT + b_t * x0 + c_t * noise

        _, denoised = self.denoise(
            model,
            x_t,
            sigmas,
            sar=sar,
            opt=opt
        )

        weights = self.get_weightings(sigmas)
        weights =  append_dims((weights), x0.ndim)
        terms["xs_mse"] = mean_flat((denoised - x0) ** 2).mean()
        terms["mse"] = mean_flat(weights * (denoised - x0) ** 2).mean()

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms


    def denoise(self, model, x_t, sigmas, sar, **model_kwargs):
        sar = sar.to(self.dtype)
        opt = model_kwargs['opt'].to(self.dtype)
        c_skip, c_out, c_in = [
            append_dims(x, opt.ndim).to(self.dtype)
            for x in self.get_bridge_scalings(sigmas)
        ]

        opt_in = c_in * x_t
        rescaled_t = (1000 * 0.25 * th.log(sigmas + 1e-44)).to(self.dtype)
        model_output = model(opt_in, rescaled_t, opt=opt_in, sar=sar).to(self.dtype)
        model_output = model_output.clamp(-100, 100)
        denoised     = c_out * model_output + c_skip * x_t
        return model_output, denoised
   

# diffusion, model, x_t, x_0
def karras_sample(
    diffusion,
    model,
    x_t,
    x_0,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
):
    assert sampler in ["heun", ], 'only heun sampler is supported currently'
    
    opt, sar = model_kwargs['opt'], model_kwargs['sar']
    xT = (opt, sar)

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)

    sample_fn = {
        "heun": partial(sample_heun, 
                        beta_d=diffusion.beta_d, 
                        beta_min=diffusion.beta_min),
    }[sampler]

    sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
        )
    
    def denoiser(x_t, sigma, sar):
        model_kwargs_no_sar = {k: v for k, v in model_kwargs.items() if k != 'sar'}
        _, denoised = diffusion.denoise(model, x_t, sigma, sar=sar, **model_kwargs_no_sar)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised
    
    x_0, path, nfe = sample_fn(
        denoiser,
        x_t,
        sigmas,
        xT=xT,
        progress=progress,
        callback=callback,
        guidance=guidance,
        **sampler_args,
    )

    print('nfe:', nfe)
    print("sample x_0 min/max:", x_0.min().item(), x_0.max().item())

    return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_bridge_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, eps=1e-4, device="cpu"):
    
    sigma_t_crit = sigma_max / np.sqrt(2)
    min_start_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_t_crit ** (1 / rho)
    sigmas_second_half = (max_inv_rho + th.linspace(0, 1, n//2 ) * (min_start_inv_rho - max_inv_rho)) ** rho
    sigmas_first_half = sigma_max - ((sigma_max - sigma_t_crit)  ** (1 / rho) + th.linspace(0, 1, n - n//2 +1 ) * (eps  ** (1 / rho)  - (sigma_max - sigma_t_crit)  ** (1 / rho))) ** rho
    sigmas = th.cat([sigmas_first_half.flip(0)[:-1], sigmas_second_half])
    sigmas_bridge = sigmas**2 *(1-sigmas**2/sigma_max**2)
    return append_zero(sigmas_bridge).to(device)


def to_d(x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
    grad_pxTlxt = (x_T - x) / (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    gt2 = 2*sigma
    d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d


def get_d_vp(x, denoised, x_T, std_t, logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp().clamp(min=1e-8, max=1e8)
    b_t = -th.expm1(logsnr_T - logsnr_t).clamp(min=1e-8, max=1e8) * logs_t.exp().clamp(min=1e-8, max=1e8)
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
    grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    if stochastic:
        return d, gt2
    else:
        return d
    

    
@th.no_grad()
def sample_heun(
    denoiser,
    x_t,
    sigmas,
    xT,
    pred_mode='vp',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    """Deterministic Heun sampler for ODE"""
    """
    sampling with ODE. -> OPTICAL IMAGE!!!!!!!!!!!!!!!!!!!!!
    """

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    opt_T, sar = xT
    x = opt_T
    B = x.shape[0]
    s_in = x.new_ones([B])
    path = []
    nfe = 0

    for i in range(40):
        sigma_i, sigma_j = sigmas[i], sigmas[i+1]
        sigma_hat = sigma_i + churn_step_ratio * (sigma_j - sigma_i)

        denoised1 = denoiser(x, sigma_hat * s_in, sar)
        d1 = to_d(x, sigma_hat, denoised1, opt_T, sigma_max=sigma_max, w=guidance)

        nfe += 1
        dt = sigma_j - sigma_hat

        if sigma_j == 0:
            x = x + d1 * dt
        else:
            x_mid = x + d1 * dt
            denoised2 = denoiser(x_mid, sigma_j * s_in, sar)
            d2 = denoised2 - x_mid
            x = x + 0.5 * (d1 + d2) * dt
            nfe += 1

        if callback:
            callback({'x': x, 'i': i, 'sigma': sigma_i, 'sigma_hat': sigma_hat, 'denoised': denoised1})

        print(f"[{i}] x min/max: {x.min().item()} / {x.max().item()}")
        print(f"[{i}] denoised1 min/max: {denoised1.min().item()} / {denoised1.max().item()}")
        print(f"[{i}] d1 min/max: {d1.min().item()} / {d1.max().item()}")
        print(f"[{i}] dt: {dt.item()}")
        print(f"[{i}] sigma_i: {sigma_i.item()}, sigma_j: {sigma_j.item()}, sigma_hat: {sigma_hat.item()}")

    return x, path, nfe

@th.no_grad()
def forward_sample(
    x0,
    y0,
    sigma_max,
    ):

    ts = th.linspace(0, sigma_max, 120)
    x = x0
    # for t, t_next in zip(ts[:-1], ts[1:]):
    #     grad_pxTlxt = (y0 - x) / (append_dims(th.ones_like(ts)*sigma_max**2, x.ndim) - append_dims(t**2, x.ndim))
    #     dt = (t_next - t) 
    #     gt2 = 2*t
    #     x = x + grad_pxTlxt * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
    path = [x]
    for t in ts:
        std_t = th.sqrt(t)* th.sqrt(1 - t / sigma_max)
        mu_t= t / sigma_max * y0 + (1 - t / sigma_max) * x0
        xt = (mu_t +  std_t * th.randn_like(x0) )
        path.append(xt)

    path.append(y0)

    return path

