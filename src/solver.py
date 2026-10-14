import torch

@torch.no_grad()
def lp_lora_threshold_solver(s, t, M=10.0, lam=0.01, p=0.5, eps=1e-12):

    swapped = s > t
    u = torch.minimum(s, t).clamp(min=eps)
    v = torch.maximum(s, t).clamp(min=eps)
    ratio_sq = (u / v).pow(2)

    # Reshape u, v, and ratio_sq into columns of shape [N, 1] 
    # so they can broadcast against 500 grid points at once
    u_exp = u.unsqueeze(1)
    v_exp = v.unsqueeze(1)
    ratio_exp = ratio_sq.unsqueeze(1)


    def objective(a_u, b_v):
        pull_1 = (u_exp.pow(2)) * ((a_u - 1.0).pow(2)) + (v_exp.pow(2)) * ((b_v - 1.0).pow(2))
        pull_1 = M * pull_1 / 2.0
        pull_2 = lam * ((u_exp * v_exp * a_u * b_v).clamp(min=0.0)).pow(p)
        return pull_1 + pull_2

    # Baseline cost if we KILL the brick: set a_u = 0, b_v = 1 -> Cost is 0.5 * M * u^2
    best_cost = 0.5 * M * u.pow(2)
    best_a_u = torch.zeros_like(u)
    best_b_v = torch.ones_like(v)


    # 1. Create 500 test values for a_u between 0 and 1 (shape: [1, 500])
    grid = torch.linspace(1e-5, 1.0 - 1e-5, steps=500, device=s.device, dtype=s.dtype).unsqueeze(0)

    # 2. Calculate the square root part of Eq. (20): sqrt(1 - 4 * a * (1 - a) * (u/v)^2)
    disc = (1.0 - 4.0 * grid * (1.0 - grid) * ratio_exp).clamp(min=0.0)
    sqrt_disc = torch.sqrt(disc)

    # 3. Test both the '+' and '-' branches of the quadratic formula
    for sign in (+1.0, -1.0):
        b_v_cand = 0.5 * (1.0 + sign * sqrt_disc)  # Shape: [N, 500]
        cand_cost = objective(grid, b_v_cand)      # Shape: [N, 500]

        # Find which of the 500 grid points gave the lowest cost for each brick
        min_cost_branch, min_idx = torch.min(cand_cost, dim=1)

        # Grab the winning a_u and b_v values at that best index
        a_u_star = grid[0, min_idx]
        b_v_star = b_v_cand[torch.arange(s.numel(), device=s.device), min_idx]

        # Did this "Survive" candidate beat the "Kill" baseline cost?
        improved = min_cost_branch < best_cost
        best_cost = torch.where(improved, min_cost_branch, best_cost)
        best_a_u = torch.where(improved, a_u_star, best_a_u)
        best_b_v = torch.where(improved, b_v_star, best_b_v)


    alpha_opt = torch.where(swapped, best_b_v, best_a_u)
    beta_opt = torch.where(swapped, best_a_u, best_b_v)
    keep_mask = (alpha_opt * beta_opt) > 0.0

    return keep_mask, alpha_opt, beta_opt