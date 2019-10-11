def multi_kl(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.size()) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (torch.log(p.clamp(1e-10,1))
                 - torch.log(q.clamp(1e-10,1)))).sum(axis)

def log_normal(x, m, v):
    
    log_prob = -((x - m) ** 2 /(2 * v)) - 0.5 * torch.log(2 * math.pi * v)

    return log_prob

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def L2(estimate, target):
	
	return ((estimate - target) ** 2).sum(1).unsqueeze(1)

# action =  sample_batched['action'].to(self.device).unsqueeze(1).requires_grad_(True)
