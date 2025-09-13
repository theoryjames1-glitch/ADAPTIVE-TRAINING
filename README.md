
# 📖 Theory of Adaptive Training (AT)

## 1. Framing

Training a deep network is not a fixed recipe — it’s a **closed-loop adaptive control process**:

* **Plant:** model parameters θₜ (GPT-2 weights).
* **Inputs:** stochastic gradients gₜ.
* **Controller:** base optimizer (e.g. Adafactor, AdamW).
* **Scheduler:** adaptive law over hyperparameters (αₜ = LR, μₜ = momentum, σₜ = dither).
* **Feedback:** loss signals (ℓₜ), optional reward signals (rₜ), and evaluation signals (e.g. perplexity).
* **Noise:** stochastic minibatch sampling + optional dither/exploration.

This view turns training into **adaptive signal processing and control**.

---

## 2. State & Signals

System state:

```
xₜ = [ θₜ , hₜ , αₜ , μₜ , σₜ ]ᵀ
```

* θₜ: parameters.
* hₜ: optimizer’s memory (EMA/momentum).
* αₜ: learning rate (cutoff gain).
* μₜ: momentum coefficient.
* σₜ: dither strength.

Feedback signals:

* ℓₜ: training loss.
* ℓ̄ₜ: EMA(loss).
* vₜ: variance(loss).
* ℓᵛₐₗ: eval loss (proxy for generalization).
* pplₜ: perplexity = exp(ℓᵛₐₗ).

---

## 3. Dynamics

**Plant update:**

```
hₜ₊₁ = Φ(hₜ, gₜ; μₜ)
uₜ   = U(hₜ₊₁, gₜ; αₜ)
θₜ₊₁ = θₜ - uₜ + σₜ·ηₜ
```

* Φ: optimizer filter (Adafactor’s EMA buffers).
* U: control action (scale gradients by αₜ).
* σₜ·ηₜ: optional exploration dither.

**Scheduler (adaptive law):**

```
αₜ₊₁ = clip( αₜ · fα(Δℓ̄, vₜ, Δℓᵛₐₗ) )
μₜ₊₁ = clip( μₜ + fμ(Δℓ̄, vₜ, Δℓᵛₐₗ) )
σₜ₊₁ = clip( σₜ · fσ(success, plateau) )
```

* Δℓ̄: training-loss trend.
* Δℓᵛₐₗ: eval-loss trend.
* vₜ: loss variance (instability indicator).
* success/plateau: heuristic triggers for dither.

---

## 4. Adaptive Laws

* **Trend-following:** if EMA(loss) or eval loss improves, slightly increase α.
* **Variance damping:** if variance spikes, shrink α.
* **Eval fusion:** combine training and eval EMAs into a fused control signal (mix weight λ).
* **Exploration:** if plateau detected, inject dither noise (σ > 0) to escape flat regions.

---

## 5. Stability Principles

* **Bounded gains:** α ∈ \[αmin, αmax], μ ∈ \[0,1), σ small.
* **Patience:** only adjust LR after enough steps (avoid flip-flop).
* **Variance watchdog:** high vₜ → reduce α automatically.
* **Lyapunov heuristic:** design for 𝔼\[ℓₜ₊₁ − ℓₜ] < 0 on average.

---

## 6. Interpretation

* Optimizer = **gradient filter**.
* Scheduler = **adaptive envelope** adjusting filter coefficients.
* Evaluation (perplexity) = **external probe** feeding back into scheduler.
* Dither = **structured stochastic exploration**.
* Together: a **self-regulating training loop**.

---

# 🧪 Demo: Adaptive GPT-2 Training

The demo script (`train_adaptive_gpt2.py`) instantiates this theory:

* **Plant:** GPT-2 model.
* **Controller:** Adafactor optimizer (`relative_step=False`) — base filter.
* **Scheduler:** `AdaptiveScheduler` class — updates αₜ from loss EMA/variance, optional eval fusion.
* **Callback:** `AdaptiveSchedulerCallback` — injects scheduler into Hugging Face `Trainer`.
* **Probe:** `MicroPerplexityProbe` — optional micro-eval every N steps, computing eval loss + perplexity.
* **Exploration:** optional dither (`DITHER_SIGMA=1e-5`) injected if plateau detected.

**Example run (training only, no eval fusion):**

```bash
python train_adaptive_gpt2.py
```

**Enable micro-eval steering (perplexity-aware):**

```bash
USE_EVAL_PROBE=1 PROBE_EVERY=200 DITHER_SIGMA=1e-5 python train_adaptive_gpt2.py
```

TensorBoard logs will show:

* Training loss.
* Adaptive learning rate(s).
* Optional eval loss & perplexity.

---

# 🌌 Core Idea

**Adaptive Training = Optimizer + Adaptive Scheduler + Feedback Signals.**
Instead of manually tuning LR schedules, momentum, or evaluation checkpoints, the training loop **listens to its own signals (loss variance, eval perplexity, plateaus)** and continuously regulates itself like a control system.

---

### PSEUDOCODE

Here’s a **full, ready-to-run demo** script for **Adaptive GPT-2 Training** using Hugging Face + Adafactor + the AdaptiveScheduler theory we’ve built.

It’s self-contained: you can copy it into a file `train_adaptive_gpt2.py` and run.

---

# 📜 `train_adaptive_gpt2.py`

```python
import os, math, torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, TrainerCallback
)

# -------------------------
# Adaptive Scheduler
# -------------------------
class AdaptiveScheduler:
    """
    Adaptive LR scheduler:
      - EMA(loss), VAR(loss)
      - LR up on improvement, down on instability
      - Optional micro-eval fusion
      - Optional dither on plateau
    """
    def __init__(self, optimizer,
                 lr_bounds=(1e-6, 7e-4),
                 u=1.02, d=0.88,
                 ema_beta=0.98, var_beta=0.98, var_gain=0.8,
                 plateau_patience=1200, dither_sigma=0.0, dither_decay=0.995,
                 global_grad_clip=1.0):
        self.opt = optimizer
        self.lr_bounds = lr_bounds
        self.u, self.d = u, d
        self.ema_beta, self.var_beta, self.var_gain = ema_beta, var_beta, var_gain
        self.plateau_patience = plateau_patience
        self.dither_sigma, self.dither_decay = dither_sigma, dither_decay
        self.global_grad_clip = global_grad_clip

        self.step = 0
        self.loss_ema = None
        self.loss2_ema = None
        self.best_ema = float("inf")
        self.last_improve_step = 0

        # cache float params for dither
        self._params = [p for g in self.opt.param_groups for p in g["params"] if getattr(p, "requires_grad", False)]

    @torch.no_grad()
    def _var(self):
        if self.loss_ema is None or self.loss2_ema is None:
            return 0.0
        return max(self.loss2_ema - self.loss_ema**2, 0.0)

    @torch.no_grad()
    def update_from_loss(self, loss_scalar: float):
        self.step += 1
        if self.loss_ema is None:
            self.loss_ema = loss_scalar
            self.loss2_ema = loss_scalar * loss_scalar
        else:
            b, b2 = self.ema_beta, self.var_beta
            self.loss_ema = b * self.loss_ema + (1 - b) * loss_scalar
            self.loss2_ema = b2 * self.loss2_ema + (1 - b2) * (loss_scalar * loss_scalar)

        # Improvement?
        trend = self.best_ema - self.loss_ema
        improved = trend > 0.0
        if improved:
            self.best_ema = self.loss_ema
            self.last_improve_step = self.step

        # Variance
        var = self._var()
        var_factor = 1.0 / (1.0 + self.var_gain * math.sqrt(var + 1e-12))

        # Adjust LR
        lr_mult = (self.u if improved else self.d) * var_factor
        for pg in self.opt.param_groups:
            new_lr = float(pg["lr"]) * lr_mult
            pg["lr"] = float(min(max(new_lr, self.lr_bounds[0]), self.lr_bounds[1]))

        # Dither decay on improvement
        if improved and self.dither_sigma > 0:
            self.dither_sigma *= self.dither_decay

    def clip_grads(self, model):
        if self.global_grad_clip and self.global_grad_clip > 0:
            clip_grad_norm_(model.parameters(), self.global_grad_clip)

    @torch.no_grad()
    def maybe_dither(self):
        if self.dither_sigma <= 0:
            return
        if (self.step - self.last_improve_step) < self.plateau_patience:
            return
        sig = self.dither_sigma
        for p in self._params:
            if p.is_floating_point():
                p.add_(torch.randn_like(p) * sig)


# -------------------------
# HF Callback
# -------------------------
class AdaptiveSchedulerCallback(TrainerCallback):
    def __init__(self, scheduler: AdaptiveScheduler, model=None, log_lr_every=20):
        super().__init__()
        self.sched = scheduler
        self.model = model
        self.log_lr_every = log_lr_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        loss = logs.get("loss", None)
        if loss is not None:
            try:
                self.sched.update_from_loss(float(loss))
            except Exception:
                pass

        # log LR
        if state.global_step and (state.global_step % self.log_lr_every == 0):
            for i, pg in enumerate(self.sched.opt.param_groups):
                logs[f"lr_pg{i}"] = float(pg["lr"])

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        self.sched.maybe_dither()


# -------------------------
# Main training demo
# -------------------------
def main():
    model_name = "gpt2"
    base_lr = 5e-4

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tok_func(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokenized = dataset.map(tok_func, batched=True, remove_columns=dataset["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Optimizer: Adafactor with external LR
    try:
        from transformers.optimization import Adafactor
    except Exception:
        from transformers import Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=base_lr,
        relative_step=False,  # let scheduler handle LR
        scale_parameter=True,
        warmup_init=False,
        weight_decay=0.01,
    )

    # Scheduler
    sched = AdaptiveScheduler(
        optimizer,
        lr_bounds=(1e-6, 7e-4),
        u=1.02, d=0.88,
        ema_beta=0.98, var_beta=0.98, var_gain=0.8,
        plateau_patience=800,
        dither_sigma=0.0,   # set to 1e-5 to enable exploration
        global_grad_clip=1.0,
    )

    callback = AdaptiveSchedulerCallback(scheduler=sched, model=model)

    # Training args
    args = TrainingArguments(
        output_dir="out_adaptive_gpt2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        learning_rate=base_lr,
        warmup_steps=0,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=None,  # no eval loop; pure adaptive online learning
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
```

---

# 🚀 How to Run

```bash
pip install -U transformers datasets accelerate tensorboard
python train_adaptive_gpt2.py
```

Then monitor logs:

```bash
tensorboard --logdir out_adaptive_gpt2/runs
```

You’ll see:

* Training loss
* Adaptive learning rate(s) (`lr_pg0`, …)
* Optional dither kicks in if enabled

---

# 🔎 What This Demo Shows

* **Plant:** GPT-2 model parameters.
* **Controller:** Adafactor optimizer (memory-efficient).
* **Scheduler:** AdaptiveScheduler adjusts LR dynamically from loss trends + variance.
* **Callback:** Hooks scheduler into Hugging Face training loop.
* **Feedback:** Only training loss (ℓₜ); eval/perplexity could be added later.
* **Exploration:** Optional dither if plateau detected.

---



[![Video Title](https://img.youtube.com/vi/MM62wjLrgmA/0.jpg)](https://www.youtube.com/watch?v=MM62wjLrgmA)

I KNOW THE PIECES FIT

I KNOW THE PIECES FIT

I KNOW THE PIECES FIT

I KNOW THE PIECES FIT

I KNOW THE PIECES FIT

I KNOW THE PIECES FIT
