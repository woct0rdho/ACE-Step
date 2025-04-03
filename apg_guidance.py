import torch


class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
    dims=[-1, -2],
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def apg_forward(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=[-1, -2],
):
    diff = pred_cond - pred_uncond
    # orig_cfg_guided = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    # print("======== 新的一轮 =========")
    # print("原来的diff", "min:", diff.min(), "max:", diff.max(), "mean:", diff.mean(), "std:", diff.std(), f"cfg会乘上{guidance_scale=}")
    # print("如果跑cfg orig_cfg_guided", "min:", orig_cfg_guided.min(), "max:", orig_cfg_guided.max(), "mean:", orig_cfg_guided.mean(), "std:", orig_cfg_guided.std())
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
        # print("跑完momentum_buffer后", "min:", diff.min(), "max:", diff.max(), "mean:", diff.mean(), "std:", diff.std(), f"cfg会乘上{guidance_scale=}")

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        # print("diff_norm", diff_norm)
        # 只有比1大的时候（爆音）才会进行缩放
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
        # print("跑完norm_threshold scale factor后", "min:", diff.min(), "max:", diff.max(), "mean:", diff.mean(), "std:", diff.std())

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims)
    # print("跑完project后, diff_parallel", "min:", diff_parallel.min(), "max:", diff_parallel.max(), "mean:", diff_parallel.mean(), "std:", diff_parallel.std())
    normalized_update = diff_orthogonal + eta * diff_parallel
    # print("跑完normalized_update后", "min:", normalized_update.min(), "max:", normalized_update.max(), "mean:", normalized_update.mean(), "std:", normalized_update.std())
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    # print("最终pred_guided", "min:", pred_guided.min(), "max:", pred_guided.max(), "mean:", pred_guided.mean(), "std:", pred_guided.std())
    return pred_guided


def cfg_forward(cond_output, uncond_output, cfg_strength):
    return uncond_output + cfg_strength * (cond_output - uncond_output)
