import functools
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from data.sft_dataset import SFTDataset
from utils.collate import sft_collate

# 指定预训练模型路径
model_path = r'D:\work\models\Meta-Llama-3.1-8B-Instruct'

# 初始化分词器，不使用 fast 版本以兼容 apply_chat_template
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config
)

# 构建 LoRA 配置
peft_config = LoraConfig(
    r=8,
    target_modules=[
        'q_proj','v_proj','k_proj','o_proj',
        'gate_proj','down_proj','up_proj'
    ],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)
# 应用 LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to('cuda')

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters())

# 构造 collate_fn
collate_fn = functools.partial(
    sft_collate,
    tokenizer=tokenizer,
    end_str='<|start_header_id|>assistant<|end_header_id""|>\n\n',
    max_length=50
)

# 加载数据集
dataset = SFTDataset('./data/sft_data.json', tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 训练循环
epochs = 10
for epoch in range(epochs):
    for inputs, loss_mask in loader:
        # 迁移到 GPU
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        loss_mask = loss_mask.to('cuda')

        # 前向计算并去掉最后一个预测
        logits = model(**inputs).logits[:, :-1, :]
        labels = inputs['input_ids'][:, 1:]

        # 形状变换
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        loss_mask = loss_mask.reshape(-1)

        # 计算交叉熵并应用掩码
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        loss = (loss * loss_mask).mean()

        # 反向传播与优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch+1} Loss: {loss.item()}')
