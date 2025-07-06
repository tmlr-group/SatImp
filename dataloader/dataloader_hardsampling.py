import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module_base import get_batch_loss
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
import pdb
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from deepspeed.utils import safe_get_full_fp32_param
from transformers.utils import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

def fc_i(name):
        class FC(nn.Module):
            def __init__(self):
                super(FC, self).__init__()
        # 4096 to llama: 32000 phi:51200
                if name=='phi':
                    self.fc = nn.Linear(2048, 51200)
                else:
                    self.fc = nn.Linear(4096, 32000)
            #self.fc = nn.Linear(2048, 51200)
            def forward(self, x):
                return self.fc(x)
        model=FC()
        return model

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):# the first ti
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.org_ckpt=kwargs.pop('ckpt_org')
        self.hyper_param=kwargs.pop('hyper_param')
        self.beta_param = kwargs.pop('beta_param')
        self.coeff = kwargs.pop('coeff')
        self.soft_param=kwargs.pop('soft_param')
        self.max_steps=kwargs.pop('max_steps')
        self.count_step=0
        self.name=kwargs.pop('model_family')
        self.log_dir = kwargs.pop('log_dir')
        self.rmu_noise=torch.rand((1,1,4096)).cuda()
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        try:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
        except: 
            self.oracle_model = self.oracle_model.cuda()

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        def get_batch_loss(output, labels):
            shifted_labels = labels[..., 1:].contiguous()
            output = output[..., :-1, :].contiguous()

            loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            # get the sum loss for each sequence in a batch
            loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

            return loss
        
        def get_batch_loss_mask(logits,labels,mask_):
            shift_logits = logits[..., :-1, :].contiguous()
            labels = labels.to(outputs.logits.device)
            shift_labels= labels[..., 1:].contiguous()
            #shift_key = mask_[..., 1:].contiguous()
            f_ce=nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = (mask_ * f_ce).sum(dim=-1)
            return loss
        
        if self.loss_type == "ga":
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            loss = forget_loss * -1

            self.count_step+=1
        
        if self.loss_type == "gd":
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            #print('Forget Loss: ', forget_loss, 'Retain Loss: ', retain_loss)
            loss = forget_loss + retain_loss

            self.count_step+=1

        if self.loss_type=='ga_topk':
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.15
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.beta)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] > threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            loss=(-1*mask_fg*fg_ce).sum()

        if self.loss_type == "ga_bottomk":
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.5
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.beta)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] < threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            loss=(-1*mask_fg*fg_ce).sum()

        if self.loss_type == "ga_rand":
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.5
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                new_matrix=(weight_split_fg[i]>0).int()
                indices = (new_matrix == 1).nonzero(as_tuple=True)[0]
                num_to_change = int(len(indices) * self.beta)
                if num_to_change > 0:
                    change_indices = indices[torch.randperm(len(indices))[:num_to_change]]  
                    #new_matrix[i][change_indices] = 0  
                    new_matrix[change_indices] = self.soft_param  
                    #pdb.set_trace()
                    remaining_indices = indices[~torch.isin(indices, change_indices)]
                    new_matrix[remaining_indices] = 1 - self.soft_param  
                spl_fg.append(new_matrix)
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{new_matrix.tolist()}\n")
            mask_fg=torch.cat(spl_fg,dim=0)
            loss=(-1*mask_fg*fg_ce).sum()

        if self.loss_type == "gd_topk":
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.15
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.beta)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] > threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss=-1*mask_fg*fg_ce
            #forget_loss=-1*fg_ce

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_labels = retain_labels.to(retain_outputs.logits.device)
            retain_shift_logits= retain_outputs.logits[..., :-1, :].contiguous()
            retain_shift_labels= retain_labels[..., 1:].contiguous()
            re_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(retain_shift_logits.view(-1, retain_shift_logits.size(-1)), retain_shift_labels.view(-1))
            weight_re = re_ce.detach()
            weight_split_re=torch.split(weight_re,499)
            spl_re=[]
            for i in range (len(weight_split_re)):
                non_zero_tensor = weight_split_re[i][weight_split_re[i] > 0]
                if len(non_zero_tensor) > 0:
                    threshold_re = torch.quantile(non_zero_tensor, self.beta)
                else:
                    threshold_re = float('inf')
                result = (weight_split_re[i] > threshold_re).int()
                with open(f'{self.log_dir}/retain_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_re.append(result) 
            mask_re= torch.cat(spl_re,dim=0)
            retain_loss= re_ce
            loss = forget_loss.sum() + retain_loss.sum()

        if self.loss_type == "gd_bottomk":
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.5
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.beta)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] < threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss=-1*mask_fg*fg_ce
            #forget_loss=-1*fg_ce

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_labels = retain_labels.to(retain_outputs.logits.device)
            retain_shift_logits= retain_outputs.logits[..., :-1, :].contiguous()
            retain_shift_labels= retain_labels[..., 1:].contiguous()
            re_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(retain_shift_logits.view(-1, retain_shift_logits.size(-1)), retain_shift_labels.view(-1))
            weight_re = re_ce.detach()
            weight_split_re=torch.split(weight_re,499)
            spl_re=[]
            for i in range (len(weight_split_re)):
                non_zero_tensor = weight_split_re[i][weight_split_re[i] > 0]
                if len(non_zero_tensor) > 0:
                    threshold_re = torch.quantile(non_zero_tensor, self.beta)
                else:
                    threshold_re = float('inf')
                result = (weight_split_re[i] > threshold_re).int()
                with open(f'{self.log_dir}/retain_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_re.append(result) 
            mask_re= torch.cat(spl_re,dim=0)
            #pdb.set_trace()
            #retain_loss= mask_re * re_ce
            retain_loss= re_ce
            loss = forget_loss.sum() + retain_loss.sum()

        if self.loss_type == "gd_rand":
            if self.hyper_param != 'None':
                self.beta = self.hyper_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                new_matrix=(weight_split_fg[i]>0).int()
                indices = (new_matrix == 1).nonzero(as_tuple=True)[0]
                num_to_change = int(len(indices) * self.beta)
                if num_to_change > 0:
                    change_indices = indices[torch.randperm(len(indices))[:num_to_change]]  
                    new_matrix[change_indices] = self.soft_param  
                    remaining_indices = indices[~torch.isin(indices, change_indices)]
                    new_matrix[remaining_indices] = 1 - self.soft_param  
                spl_fg.append(new_matrix)
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{new_matrix.tolist()}\n")
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss=-1*mask_fg*fg_ce

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_labels = retain_labels.to(retain_outputs.logits.device)
            retain_shift_logits= retain_outputs.logits[..., :-1, :].contiguous()
            retain_shift_labels= retain_labels[..., 1:].contiguous()
            re_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(retain_shift_logits.view(-1, retain_shift_logits.size(-1)), retain_shift_labels.view(-1))
            weight_re = re_ce.detach()
            weight_split_re=torch.split(weight_re,499)
            spl_re=[]
            for i in range (len(weight_split_re)):
                new_matrix_re=(weight_split_re[i]>0).int()
                indices_re = (new_matrix_re==1).nonzero(as_tuple=True)[0]
                num_to_change_re = int(len(indices_re) * self.beta)
                if num_to_change_re > 0:
                    change_indices_re = indices_re[torch.randperm(len(indices_re))[:num_to_change_re]]  
                    new_matrix_re[change_indices_re] = self.soft_param  
                    remaining_indices_re = indices_re[~torch.isin(indices_re, change_indices_re)]
                    new_matrix_re[remaining_indices_re] = 1 - self.soft_param 
                with open(f'{self.log_dir}/retain_record.txt','a') as file:
                    file.write(f"{new_matrix_re.tolist()}\n")
                spl_re.append(new_matrix_re)
            mask_re=torch.cat(spl_re,dim=0)
            #mask_re=new_matrix_re.view(-1)
            #pdb.set_trace()
            #retain_loss= mask_re * re_ce
            retain_loss= re_ce
            loss = forget_loss.sum() + retain_loss.sum()
        
        if self.loss_type == "npo_rand":
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                new_matrix=(weight_split_fg[i]>0).int()
                indices = (new_matrix == 1).nonzero(as_tuple=True)[0]
                num_to_change = int(len(indices) * self.hyper_param)
                if num_to_change > 0:
                    change_indices = indices[torch.randperm(len(indices))[:num_to_change]]  
                    new_matrix[change_indices] = self.soft_param  
                    remaining_indices = indices[~torch.isin(indices, change_indices)]
                    new_matrix[remaining_indices] = 1 - self.soft_param  
                spl_fg.append(new_matrix)
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{new_matrix.tolist()}\n")
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

        if self.loss_type == "npo_bottomk":
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.hyper_param)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] < threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

        if self.loss_type=='npo_topk':
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.hyper_param)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] > threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

        if self.loss_type == "npo_gd_rand":
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                new_matrix=(weight_split_fg[i]>0).int()
                indices = (new_matrix == 1).nonzero(as_tuple=True)[0]
                num_to_change = int(len(indices) * self.hyper_param)
                if num_to_change > 0:
                    change_indices = indices[torch.randperm(len(indices))[:num_to_change]] 
                    new_matrix[change_indices] = self.soft_param  
                    remaining_indices = indices[~torch.isin(indices, change_indices)]
                    new_matrix[remaining_indices] = 1 - self.soft_param  
                spl_fg.append(new_matrix)
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{new_matrix.tolist()}\n")
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.coeff * forget_loss + retain_loss

        if self.loss_type=='npo_gd_topk':
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.hyper_param)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] > threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.coeff * forget_loss + retain_loss
        
        if self.loss_type == "npo_gd_bottomk":
            if self.beta_param != 'None':
                self.beta = self.beta_param
            else: self.beta=0.2
            forget_inputs, retain_inputs = inputs
            #print(len(forget_inputs),len(retain_inputs))
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels= labels[..., 1:].contiguous()
            fg_ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weight_fg=fg_ce.detach()
            weight_split_fg=torch.split(weight_fg,499)
            spl_fg=[]
            for i in range (len(weight_split_fg)):
                non_zero_tensor = weight_split_fg[i][weight_split_fg[i] > 0]
                #print(non_zero_tensor)
                if len(non_zero_tensor) > 0:
                    threshold = torch.quantile(non_zero_tensor, 1-self.hyper_param)
                else:
                    threshold = float('inf')
                result = (weight_split_fg[i] < threshold).int()
                with open(f'{self.log_dir}/forget_record.txt','a') as file:
                    file.write(f"{non_zero_tensor.tolist()}\n")
                spl_fg.append(result)
            mask_fg=torch.cat(spl_fg,dim=0)
            forget_loss_current=(mask_fg*fg_ce).sum(dim=-1)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss_mask(forget_logits_oracle, labels, mask_fg)
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.coeff * forget_loss + retain_loss

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 
                # if 'eval_log' not in eval_task:
                #     normalize_gt = True

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def custom_data_collator_forget_key_human(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        mask = [s[3] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(mask)))
    return rets

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
