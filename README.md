# [ICML2025] Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning

This is the code for ICML2025 paper: [Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning](https://arxiv.org/abs/2505.11953)

## Installation
```python
conda create -n unlearning python=3.10
conda activate unlearning
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset
To load the dataset, use the following code:
```python
# For the TOFU benchmark
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU","full")

# For the MUSE benchmark
SUBSET = "verbmem"
SPLIT = "forget"
dataset = load_dataset("muse-bench/MUSE-Books", SUBSET, split=SPLIT)

# For the WMDP benchmark, please refer to the official requirements in [wmdp.ai](https://www.wmdp.ai/)
```
1. The proposed important token annotation on TOFU is presented in `data/importance_forget**.pth`, which can be utilized convinently with proposed code in `dataloader/data_module_base.py`

# Training and Evaluation
2. We present some samples to use this repo:

```python
#For forget_base.py, beta controls the smoothness of weight distribution
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 forget_base.py --config-name=forget_base.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=grad_ascent weight_decay=0.0 beta=3.0 npo_coeff=0.2
#For forget_granu.py, hyper_param is the beta to control the smoothness of weight distribution
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 forget_granu.py --config-name=forget_granu.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=instance_simsat_ga weight_decay=0.0 hyper_param=3.0 
# For forget_hardsampling.py, hyper_param is the beta to control the allocation of weights
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 forget_hardsampling.py --config-name=forget_hardsampling.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=ga_topk weight_decay=0.0 hyper_param=0.3
# For forget_imp.py, beta is the p to control allocation
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 forget_imp.py --config-name=forget_imp.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=ga_topk weight_decay=0.0 beta=0.3
# For SatImp, hyper_param is beta1, beta is beta2. For SimSat and SimImp, beta controls the weight distribution.
# The recommended hyper-parameters of SatImp method is beta1=5.0, beta2=0.1
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 forget_sat.py --config-name=forget_sat.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=satimp weight_decay=0.0 beta=0.1 hyper_param=5.0

# For WMDP and MUSE, the unlearning is similiar
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=27393 muse_base.py --config-name=muse_base.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=simnpo split=news weight_decay=0.0 beta=3.0 npo_coeff=0.2 
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=27393 wmdp_base.py --config-name=wmdp_base.yaml batch_size=2 gradient_accumulation_steps=8 forget_loss=grad_ascent weight_decay=0.0


# Evaluation with ES score, please modify the model_path to your checkpoints
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=18149 eval_es.py model_family=llama2-7b split=forget01 model_path=*/llama2-7b/method_1e-05_forget01_8_0.0_2_0.1/checkpoint-25 ps_type=exact
```
