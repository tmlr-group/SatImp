# an example: 
# os.system('CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=18149 eval_es.py model_family=llama2-7b split=forget01 model_path=*/llama2-7b/method_1e-05_forget01_8_0.0_2_0.1/checkpoint-25 ps_type=similar')
# note: add a property of 'ps_type' in config/eval_everything.yaml, taking values from exact, perturb, and similar
import pdb, os, hydra
import logging
import random,time,zlib
import numpy as np
import sklearn.metrics as sk

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from utils import get_model_identifiers_from_yaml


import safetensors
log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def model_mix(model,before,after,update_ratio):
    for name,parameter in model.named_parameters():
        parameter.data=update_ratio*before[name[:]].cuda()+(1-update_ratio)*after[name[:]].cuda()
    return model    


@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                except:
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                except:
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    
    if model_id=='microsoft/phi-1_5':
        before_ckpt=safetensors.torch.load_file('../tofu/data/weight/ft_epoch5_lr2e-05_phi_full_wd0.0/checkpoint-625/model.safetensors')
        after_ckpt=safetensors.torch.load_file(cfg.model_path+'/model.safetensors')
    if model_id=='NousResearch/Llama-2-7b-chat-hf':
        before_ckpt_1=safetensors.torch.load_file('../tofu/data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00001-of-00003.safetensors')
        before_ckpt_2=safetensors.torch.load_file('../tofu/data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00002-of-00003.safetensors')
        before_ckpt_3=safetensors.torch.load_file('../tofu/data/weight/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00003-of-00003.safetensors')
        before_ckpt={**before_ckpt_1,**before_ckpt_2,**before_ckpt_3}
        after_ckpt1=safetensors.torch.load_file(cfg.model_path+'/model-00001-of-00003.safetensors')
        after_ckpt2=safetensors.torch.load_file(cfg.model_path+'/model-00002-of-00003.safetensors')
        after_ckpt3=safetensors.torch.load_file(cfg.model_path+'/model-00003-of-00003.safetensors')
        after_ckpt={**after_ckpt1,**after_ckpt2,**after_ckpt3}
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    
    if cfg.split=='forget10':
        retain_name='retain90'
    elif  cfg.split=='forget05':
        retain_name='retain95'
    elif  cfg.split=='forget01':
        retain_name='retain99'
    
    if cfg.ps_type == 'perturb':
        
        if cfg.split=='forget01':
            forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train']
            retain_eval_data=load_dataset('locuslab/TOFU','retain_perturbed')['train']
        elif cfg.split=='forget05':
            forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train']
            retain_eval_data=load_dataset('locuslab/TOFU','retain_perturbed')['train']
        else:
            forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train']
            retain_eval_data=load_dataset('locuslab/TOFU','retain_perturbed')['train']
    else:
        if cfg.split=='forget01':
            forget_data=load_dataset('locuslab/TOFU',cfg.split)['train']
            retain_eval_data=load_dataset('locuslab/TOFU',retain_name)['train'].train_test_split(train_size=400,shuffle=False)['train']
        elif cfg.split=='forget05':
            forget_data=load_dataset('locuslab/TOFU',cfg.split)['train']
            retain_eval_data=load_dataset('locuslab/TOFU',retain_name)['train'].train_test_split(train_size=400,shuffle=False)['train']
        else:
            forget_data=load_dataset('locuslab/TOFU',cfg.split)['train']
            retain_eval_data=load_dataset('locuslab/TOFU',retain_name)['train'].train_test_split(train_size=400,shuffle=False)['train']
        
        
    retain_eval_loader=torch.utils.data.DataLoader(retain_eval_data,batch_size=1)
    forget_loader=torch.utils.data.DataLoader(forget_data,batch_size=1)

    log1 = logging.getLogger("Unlearning")
    if cfg.ps_type == 'perturb':
        log_file_path = cfg.model_path+f'/ps_perturb.log'
    elif cfg.ps_type == 'exact': 
        log_file_path = cfg.model_path+f'/ps_exact.log'
    else: raise RuntimeError('error here in logger')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    log1.addHandler(file_handler)
    
    def string2token(strings):
        tks = [tokenizer.encode(_, add_special_tokens=True, return_tensors='pt').to(model.device)[0] for _ in strings]
        tk_lens = [_.size(0) for _ in tks]
        return {'token': tks, 'length': tk_lens}
    def token2string(tokens):
        strs = [tokenizer.decode(_, skip_special_tokens=True) for _ in tokens]
        return strs

    def lcs(s1,s2):
        a = [[None for i in range(len(s2))] for j in range(len(s1))]
        def _lcs(s1, s2, s1Index, s2Index, arr):
            if s1Index ==-1 or s2Index == -1:
                return 0
            if(arr[s1Index][s2Index] != None):
                return arr[s1Index][s2Index]
            if s1[s1Index] == s2 [s2Index]:
                result = 1+ _lcs(s1, s2, s1Index -1, s2Index -1, arr)
            else:
                result= max(_lcs(s1, s2, s1Index -1, s2Index, arr), _lcs(s1, s2, s1Index, s2Index -1, arr))
            arr[s1Index][s2Index] = result
            return result 
        return _lcs(s1, s2, len(s1)-1, len(s2)-1, a)

    def processing(loader, model):
        ps_list = []
        for idx, s in enumerate(loader):
            if cfg.ps_type == 'perturb':
                ques, anws = s['paraphrased_question'], s['answer']
            else:
                ques, anws = s['question'], s['answer'] # ['paraphrased_question'][0]
            fuls = [f"### Question: {que}\n ### Answer: {ans}" for que, ans in zip(ques, anws)]
            ques = [f"### Question: {que}\n ### Answer: " for que, ans in zip(ques, anws)]
            _ques_tks_and_lens, _fuls_tks_and_lens = string2token(ques), string2token(fuls)
            ques_tks, ques_tks_lens = _ques_tks_and_lens['token'], _ques_tks_and_lens['length']
            fuls_tks, fuls_tks_lens = _fuls_tks_and_lens['token'], _fuls_tks_and_lens['length']
            #if cfg.zlib_set==True:
            #    zlib_entropy=len(zlib.compress(bytes(s['answer'][0], "utf-8")))
            outputs=model.model(fuls_tks[0].unsqueeze(0))
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
            shift_logits=logits[:,ques_tks_lens[0]-1:-1]
            pred_tks_=shift_logits.argmax(dim=2).squeeze(0).flip(dims=[0])
            fuls_tks_ = fuls_tks[0][ques_tks_lens[0]:].flip(dims=[0])
            #pdb.set_trace()
            same=0
            for i in range (len(pred_tks_)):
                if pred_tks_[i].item()==fuls_tks_[i].item():
                    same+=1
                else:
                    break   
            ps_list += [same/len(pred_tks_)]
        return ps_list
    model=model_mix(model,before_ckpt,after_ckpt,0)
    ps_forget = processing(forget_loader, model)
    ps_forget = sum(ps_forget) / len(ps_forget)
    ps_retain = processing(retain_eval_loader, model)
    ps_retain = sum(ps_retain) / len(ps_retain)
    log1.info('unlearned model: ps retain %.4f forget %.4f | retain bar %.4f' 
              % (ps_retain, ps_forget, ps_retain * cfg.ps_p))
    
    model=model_mix(model,before_ckpt,after_ckpt,1)
    ps_forget = processing(forget_loader, model)
    ps_forget = sum(ps_forget) / len(ps_forget)
    ps_retain = processing(retain_eval_loader, model)
    ps_retain = sum(ps_retain) / len(ps_retain)
    log1.info('original model: ps retain %.4f forget %.4f | retain bar %.4f' 
              % (ps_retain, ps_forget, ps_retain * cfg.ps_p))
    
if __name__ == "__main__":
    main()