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


class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t
    

class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(th.DoubleTensor([1])).item()

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_f_g2(self, t):
        t = t.to(th.float32)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(th.float32)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T

        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t
    

class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings
    

class DDBMPreCond(PreCond):
    def __init__(self, ns, sigma_data, cov_xy):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        self.sigma_data_end = sigma_data

    def _get_scalings_and_weightings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)

        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        c_noise = 1000 * 0.25 * th.log(t + 1e-44)
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class KarrasDenoiser(nn.Module):
    """
    Diffusion bridge denoiser for VE/VP preds with Karras schedule.
    """
    def __init__(
        self,
        sigma_data = 0.5,
        sigma_max= 1.0,
        sigma_min = 0.0001,
        beta_d = 2,
        beta_min = 0.1,
        cov_xy = 0.0,
        rho = 7.0,
        pred_mode = 'vp',
        weight_schedule = 'karras',
        loss_norm= 'lpips',
        num_timesteps = 40,
        image_size = 256,
        dtype=th.float32,
        precond=''
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


        ns = VPNoiseSchedule(beta_d=self.beta_d, beta_min=self.beta_min)
        self.precond = DDBMPreCond(ns, sigma_data=self.sigma_data, cov_xy=self.cov_xy)
        self.noise_schedule = ns


    def bridge_sample(self, x0, xT, t, noise):
        a_t, b_t, c_t = [append_dims(item, x0.ndim) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples


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

        sigmas = th.minimum(sigmas, th.ones_like(sigmas) * self.sigma_max)
        terms = {}

        x_t = self.bridge_sample(x0, opt, sigmas, noise)
        print("x_t:", x_t.min().cpu().item())

        _, denoised, weights = self.denoise(
            model,
            x_t,
            sigmas,
            sar=sar,
            opt=opt
        )

        terms["xs_mse"] = mean_flat((denoised - x0) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x0) ** 2)

        terms["loss"] = terms["mse"]

        return terms


    def denoise(self, model, x_t, sigmas, sar, **model_kwargs):
        sar = sar.to(self.dtype)
        opt = model_kwargs['opt'].to(self.dtype)

        c_skip, c_in, c_out, c_noise, weights = self.precond.get_scalings_and_weightings(sigmas, opt.ndim)

        opt_in = c_in * x_t
        model_output = model(opt_in, c_noise, opt=opt_in, sar=sar).to(self.dtype)
        denoised     = c_out * model_output + c_skip * x_t
        print("model", model_output.min().cpu().item(), model_output.max().cpu().item())
        print("denoi",denoised.min().cpu().item(), denoised.max().cpu().item())
        print("c-skip",c_skip.min().cpu().item(), c_skip.max().cpu().item())
        print("in",c_in.min().cpu().item(), c_in.max().cpu().item())
        print("out",c_out.min().cpu().item(), c_out.max().cpu().item())
        print("noise",c_noise.min().cpu().item(), c_noise.max().cpu().item())
        print("weights",weights.min().cpu().item(), weights.max().cpu().item())
        return model_output, denoised, weights
   

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
    
    opt = model_kwargs['opt']
    sar = model_kwargs['sar']

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)
  
    def denoiser(x, sigma):
        _, denoised, _ = diffusion.denoise(model, x, sigma, opt=opt, sar=sar, x0=x_0)
        
        return denoised.clamp(-1,1)
        

    sample_fn = sample_heun
    sampler_args = dict(
            churn_step_ratio=churn_step_ratio, 
            sigma_max=sigma_max
        )
    sample, path, nfe = sample_fn(
        denoiser,
        x_t,
        sigmas,
        guidance=guidance,
        progress=progress,
        callback=callback,
        **sampler_args,
    )

    print('nfe:', nfe)
    print("sample x_0 min/max:", sample.min().item(), sample.max().item())

    return sample.clamp(-1, 1), path, nfe


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
    x,
    sigmas,
    churn_step_ratio=0.,
    guidance=1,
    progress=False,
    callback=None,
    sigma_max=80.0,
):
    
    indices = range(len(sigmas)-1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    
    path = []
    nfe = 0
    B = x.size(0)

    for i in indices:
        sigma_i, sigma_j = sigmas[i], sigmas[i+1]
        sigma_hat = sigma_i + churn_step_ratio * (sigma_j - sigma_i)
        
        denoised1 = denoiser(x, sigma_hat)
        d1 = to_d(x, sigma_hat, denoised1, x, sigma_max=sigma_max, w=guidance)
        nfe += 1

        dt = sigma_j - sigma_hat
        if sigma_j == 0:
            x = x + d1 * dt
        else:
            x_mid = x + d1 * dt
            denoised2 = denoiser(x_mid, sigma_j)
            d2 = to_d(x_mid, sigma_j, denoised2, x, sigma_max, w=guidance)
            x = x + 0.5 * (d1 + d2) * dt
            nfe += 1

        if callback:
            callback({'x': x, 'i': i, 'sigma': sigma_i, 'sigma_hat': sigma_hat, 'denoised': denoised1})
       
        x = x.clamp(-1, 1)
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

