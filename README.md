# QLoRA + Flash Attention 微调 Qwen2.5 学习项目

基于 ShareGPT 格式的医疗问答数据，使用 **QLoRA** 和 **Flash Attention 2** 对 **Qwen2.5-3B-Instruct** 模型进行指令微调，最终实现一个具备自我认知的智能医生客服机器人。

---

## 项目亮点

- 使用 **4bit 量化（QLoRA）** 大幅降低显存占用，使消费级 GPU 也能微调大模型
- 集成 **Flash Attention 2** 加速注意力计算，提升训练效率
- 融合**医疗问答数据**与**自我认知数据**，使模型具备角色扮演能力
- 基于 **LoRA** 只训练少量参数，训练速度快（约 11 分钟完成 1000 步）

---

## 项目结构

```
QLoRA-FlashAttention-Qwen2/
├── data.csv                  # 医疗问答训练数据（约 10 万条）
├── self_cognition.json       # 自我认知数据（角色：智能医生客服机器人小D）
├── QLoRA_FlaAtt.ipynb        # 主训练 Notebook
├── requirements.txt          # Python 依赖
├── checkpoints_self_cong/    # 训练过程中保存的 checkpoint
│   ├── checkpoint-500/
│   └── checkpoint-1000/
└── merged_model/             # 合并 LoRA 权重后的完整模型
```

---

## 环境配置(这个在QLoRA_FlaAtt.ipynb也有)

### 1. 创建虚拟环境

```bash
conda create --prefix ./qlora python=3.10 -y
conda activate ./qlora
```

### 2. 安装依赖

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install -r requirements.txt
pip install packaging ninja ipykernel
python -m ipykernel install --sys-prefix --name qlora --display-name "Python (qlora)"
```

### 3. 安装 Flash Attention

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.1.post1/flash_attn-2.5.1.post1+cu118torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install einops scipy sentencepiece
```

### 4. 下载基础模型

```bash
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ~/autodl-tmp/QLoRA-FlashAttention-Qwen2
```

---

## 训练流程

### 核心配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-3B-Instruct |
| 量化方式 | 4bit NF4 (QLoRA) |
| 注意力机制 | Flash Attention 2 |
| LoRA Rank | 32 |
| LoRA Alpha | 16 |
| LoRA 作用模块 | q_proj, k_proj, v_proj, o_proj |
| 学习率 | 2e-4 |
| 优化器 | paged_adamw_8bit |
| 训练步数 | 1000 步 |
| 训练时间 | 约 677 秒（~11 分钟） |

### 数据格式

训练数据统一转换为 ChatML 格式：

```json
{
  "id": "identity_0",
  "conversations": [
    {"from": "user", "value": "用户的提问"},
    {"from": "assistant", "value": "模型的回答"}
  ]
}
```

### 运行训练

打开 `QLoRA_FlaAtt.ipynb`，按顺序执行各个 Cell 即可。

---

## 推理使用

### 加载合并后的模型进行推理

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载基础模型并合并 LoRA 权重
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "checkpoints_self_cong/checkpoint-1000")
model = model.merge_and_unload()
model.eval()

# 构造推理输入（需与训练格式一致）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "我的宝宝肚子不舒服，这是怎么回事？"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
input_length = inputs["input_ids"].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# 只取新生成的部分
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response)
```

> **注意**：推理时必须使用 `apply_chat_template` 构造输入，直接传入原始文本会导致输出格式异常。

---

## 训练损失

| 步数 | 训练损失 |
|------|---------|
| 100  | 1.2957  |
| 200  | 0.6354  |
| 300  | 0.5282  |
| 500  | 0.3791  |
| 700  | 0.3711  |
| 1000 | 0.3804  |

---

## 主要依赖

- `torch==2.1.0`
- `transformers`
- `peft`
- `bitsandbytes`
- `flash-attn==2.5.1.post1`
- `trl`

---

## 学习要点

1. **QLoRA 原理**：通过 4bit 量化冻结基础模型，仅训练低秩适配层（LoRA），显著降低显存需求
2. **Flash Attention 2**：用 CUDA Kernel 优化注意力计算，提升训练速度并减少显存占用
3. **数据格式对齐**：训练与推理必须使用相同的 prompt 格式，否则模型输出会出现异常
4. **Checkpoint 选择**：保存多个 checkpoint，可根据验证集效果选择最优的模型权重
