from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

max_seq_length = 2048  # Supports RoPE Scaling internally, so choose any!

# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"

dataset = load_dataset("json", data_files={"train": url}, split="train")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",  # NEW! Llama 3.3 70B!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4B-it",
    max_seq_length=max_seq_length,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# 這裡出問題 因為Transformers版本的問題
trainer.train()

# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Customized chat templates


# model儲存====================================================
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")


# 重新載入Lora finet的模型===========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LORA_MODEL_NAME",  # 這裡要放model當初儲存的路徑位置
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)


# inference==========================================================
##demo1-----------------------

# 定義提示模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 準備輸入資料
instruction = "Continue the fibonnaci sequence."
input_text = "1, 1, 2, 3, 5, 8"

# 將提示轉換為模型輸入
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,  # instruction
            input_text,  # input
            "",  # output - leave this blank for generation!
        )
    ],
    return_tensors="pt",
).to("cuda")


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference


outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)

# demo2-------------------------------------------
# 單輪對話
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# 設定參數
max_seq_length = 2048
dtype = None  # 自動偵測最佳精度
load_in_4bit = True  # 啟用 4-bit 量化以減少記憶體使用

# 載入先前訓練好的 LoRA 模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # 替換為您的模型路徑或名稱
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 啟用 Unsloth 的快速推理模式
FastLanguageModel.for_inference(model)

# 準備單輪對話的提示
prompt = "海綿寶寶的書法是不是叫做海綿體？"

# 將提示轉換為模型輸入
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 初始化 TextStreamer 以即時顯示生成的文字
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# 生成回應
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=64)

# demo3-----------------------------------
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# 套用聊天模板（以 ChatML 為例）
tokenizer = get_chat_template(tokenizer, chat_template="chatml")


# 定義多輪對話歷史
messages = [
    {"role": "system", "content": "你是一個樂於助人的 AI 助手。"},
    {"role": "user", "content": "請用中文回答。"},
    {"role": "assistant", "content": "好的，請問有什麼我可以幫助您的？"},
    {"role": "user", "content": "海綿寶寶的書法是不是叫做海綿體？"},
]


# 應用聊天模板並轉換為模型輸入
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


# 初始化 TextStreamer 以即時顯示生成的文字
text_streamer = TextStreamer(tokenizer, skip_prompt=True)


# 生成回應
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=300)

# demo4----------------------------------------------

# 準備模擬多代理對話的提示
prompt = """
[AI1]：請針對「生成式AI未來發展前景？」這個問題，提出三個可能的回答方向。
[AI2]：根據 AI1 的建議，詳細撰寫每個回答方向的內容。
[AI3]：請檢查 AI2 的回答是否正確，並提供更好的摘要。
"""

# 將提示轉換為模型輸入
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 初始化 TextStreamer 以即時顯示生成的文字
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# 生成回應
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
