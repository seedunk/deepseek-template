import os
os.environ["TRITON_INTERPRET"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
 
model_path =os.getenv("deepseek_r1_distill_qwen_7b",  "../models/deepseek_r1_distill_qwen_7b")   
#/mnt/d/DeepSeek_R1_Distill_Qwen_7B"

tokenizer = AutoTokenizer.from_pretrained(model_path)



model_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=model_config_4bit,
    device_map="auto",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

 

def helloworld(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(chat("你好，请介绍一下自己"))