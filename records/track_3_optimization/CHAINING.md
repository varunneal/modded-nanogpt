# Optimizer Chaining Reference (modded-nanogpt track_3)

## Canonical pipeline

```
raw grad g
  [1] momentum EMA              -> m
  [2] whitening preconditioner  -> reshape in eigenbasis (rotate -> diag rescale -> rotate back)
  [3] spectral shaping          -> Newton-Schulz orthogonalization (p = 0)
  [4] direction corrections     -> Contra-/Soft-Muon blends with raw grad
  [5] postconditioner           -> NorMuon row/col variance normalization
  [6] update constraints        -> aspect-ratio scale, u/w floor, trust gate
  ---- form update, p <- p - lr*update ----
  [7] weight-space constraints  -> hyperball projection, SODA anchor (act on p, not on update)
```

Stages [2]-[6] each operate on a single matrix and restore its Frobenius norm before handoff.
Stage [7] operates on the weight `p`, not on the update direction.

## Invariants that make chaining work

1. Accumulate independently, apply in sequence. Preconditioner state (Gram matrices, Q
   factors, second moments) is updated every step from the raw grad/momentum. Application
   order is a separate concern from accumulation; multiple preconditioners can be accumulated
   simultaneously without interfering.
2. Norm preservation between stages. Every stage restores ||.||_F of its input before
   handing off, so stages commute in scale and a single final lr/constraint sets magnitude.
3. Order = increasing applied spectral power. whitening (basis rotation + diag rescale)
   before spectral (orthogonalization) before row/col postconditioning.
4. Weight-space constraints stay outside the preconditioner chain (operate on p, applied
   last), so they compose with any upstream chain unchanged.

## Spectral powers (applied update = U sigma^p V^T)

| method                       | p          |
|------------------------------|------------|
| two-sided whitening          | -1         |
| PSGD-Kron (k=1)              | -1/3       |
| Muon / 1/4-Shampoo / NorMuon | 0          |
| generalized c_k              | (k-2)/(k+2)|

Chaining = composing powers (apply lower-p stage first, renormalize, then higher-p stage).

---

## Stage mechanisms (verbatim)

### [1] Momentum EMA
```python
state["momentum"].lerp_(grad, 1 - mu)
momentum_update = grad.lerp(state["momentum"], mu)   # nesterov blend
```

### [2] Whitening preconditioner

SOAP (rotate to row/col eigenbasis, EMA second moment in basis, divide, rotate back, renorm):
```python
def soap_precondition_momentum(update, state, beta2, eps=1e-8):
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update.float() @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    precond = q_row @ (projected / state["exp_avg_sq"].sqrt().add(eps)) @ q_col.T
    precond.mul_(update.float().norm() / precond.norm().clamp_min(eps))   # norm-preserving
    return precond.to(update.dtype)
```
Accumulation (raw grad, EMA Gram; eigenbasis refreshed every precondition_frequency steps):
```python
state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - shampoo_beta)
state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - shampoo_beta)
# init: q = eigenbasis(gg); refresh: q, _ = torch.linalg.qr(gg @ q)
```

PSGD-Kron (whitening criterion, Kron factors Q0,Q1; analytic identity trick removes probe):
```python
# accumulate factors (every step), then apply:
A = Q0 @ U @ Q1.mT                                    # U = momentum
B = solve_triangular(Q0.mT, v, upper=False, left=True)
B = solve_triangular(Q1, B, upper=True, left=False)   # B = Q0^-T v Q1^-1
Q0.sub_(precond_lr/spectral_norm(A@A.mT + B@B.mT) * torch.triu(A@A.mT - B@B.mT) @ Q0)
Q1.sub_(precond_lr/spectral_norm(A.mT@A + B.mT@B) * torch.triu(A.mT@A - B.mT@B) @ Q1)
step_update = (Q0.mT @ Q0) @ update @ (Q1.mT @ Q1)    # application
```

### [3] Spectral shaping (Newton-Schulz orthogonalization)
```python
X = X / (X.norm(dim=(-2,-1), keepdim=True) + 1e-7)
a, b, c = 2, -1.5, 0.5
for _ in range(12):
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X
update *= max(1, grad.size(-2) / grad.size(-1))**0.5   # aspect-ratio scale
```

### [4] Direction corrections
Contra-Muon (add unit-operator-norm grad, renorm):
```python
normalized_grad = scale_to_unit_operator_norm(update.clone())
ns_update = zeropower_via_newtonschulz5(update)
contra_update = ns_update + contra_coeff * normalized_grad
contra_update = contra_update * ns_update.norm() / contra_update.norm()   # norm-preserving
```
Soft-Muon (blend soft orthogonalization into contra, schedule-dependent):
```python
soft_update = soft_via_newtonschulz5(update, p, scale, input_norm)
soft_update = soft_update * ns_update.norm() / soft_update.norm()
update = contra_update + (soft_update - contra_update) * blend
update = update * ns_update.norm() / update.norm()
```

### [5] Postconditioner (NorMuon row/col variance)
```python
if update.size(-2) >= update.size(-1):
    per_row_var = (update*update).mean(dim=-1, keepdim=True)
else:
    per_row_var = (update*update).mean(dim=-2, keepdim=True)
second_moment.lerp_(per_row_var.float(), 1 - beta2)
vnorm = update.norm()
update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
update = update * (vnorm / update.norm().clamp_min(1e-10))   # norm-preserving
```

### [6] Update constraints (u/w floor)
```python
p_fro = p.float().norm().clamp_min(1e-8)
u_fro = update.float().norm().clamp_min(1e-8)
cur_uw = u_fro / p_fro
scale = torch.where(cur_uw < TARGET_UW, TARGET_UW * p_fro / u_fro, torch.ones_like(p_fro))
update = update * scale.to(update.dtype)
```

### [7] Weight-space constraints
Hyperball (identical across NorMuonH / KL-SOAP-H / SOAP-H / PSGD-H; operates on p):
```python
def scale_invariant_update_(param, update, lr, eps=1e-10):
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    param.copy_(new_param / torch.clamp(new_param.norm(), min=eps) * p_norm)
```
SODA anchor (pull weight toward init, faded out by schedule; applied before the update add):
```python
lam = min(max(soda_scale / (step + 2), 0.0), 1.0)
p.mul_(1 - lam).add_(state["soda_anchor"], alpha=lam)
```

---

## Concrete chains

SinkSOAP (gram_sinkhorn_normuon_update):
```
momentum -> EMA GG^T,G^TG -> eigenbasis U,V -> A=U^T M V -> Sinkhorn(A^2)
  -> U A_bal V^T -> scale_update_like_muon -> NorMuon postcondition (norm-preserving)
```

SOAP-Muon (contra_muon_mlp_soapish), MLP params:
```
momentum -> SOAP precond -> Newton-Schulz -> Contra-Muon -> aspect scale
  -> NorMuon -> u/w floor -> p += -lr*update -> refresh SOAP basis
```

NorMuonH:
```
momentum(nesterov) -> Newton-Schulz -> aspect scale -> Adafactor row/col var precond
  -> hyperball projection (scale_invariant_update_)
```

KL-SOAP-H:
```
project grad to eigenbasis -> Adam(beta1,beta2) in basis -> project back
  -> update Gram + eigenvalue estimates -> refresh basis every precondition_frequency
  -> hyperball projection
```

contra_soft_soda (full stack), MLP/attn params:
```
momentum -> SOAP precond (trust-gate on attn.proj) -> Newton-Schulz
  -> Contra-Muon -> Soft-Muon blend -> aspect scale -> NorMuon -> u/w floor
  -> SODA anchor (on p) -> p += -lr*update -> refresh SOAP basis
```

PSGD-Kron-H:
```
momentum(bias-corrected) -> sample probes v,nu -> psgd_update_precond(Q0,Q1)
  -> step_update = (Q0^T Q0) update (Q1^T Q1) -> hyperball projection
```

## Parameter routing
- 2D block weights (attn q/k/v/proj, mlp fc/proj): Muon/SOAP/PSGD chain above.
- SOAP subset: typically mlp.fc, mlp.proj (+ attn.v / attn.proj in full stack).
- Embedding, output head, all 1D (biases, norm gains): AdamW, separate lrs.
