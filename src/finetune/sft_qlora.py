"""Optional QLoRA fine-tuning so the small model follows Themis's answer format.
Needs an NVIDIA GPU (a free Colab T4 works) and the optional packages in requirements.txt.

Run:  python -m src.finetune.make_sft_data && python -m src.finetune.sft_qlora
Then: THEMIS_ADAPTER=models/themis-lora python eval/run_eval.py
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config import LLM_MODEL, PROC_DIR, ROOT

OUT_DIR = ROOT / "models" / "themis-lora"


def main() -> None:
    data = load_dataset("json", data_files={"train": str(PROC_DIR / "sft_train.jsonl"),
                                            "validation": str(PROC_DIR / "sft_val.jsonl")})
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb, device_map="auto")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    args = SFTConfig(output_dir=str(OUT_DIR), num_train_epochs=2, per_device_train_batch_size=2,
                     gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10,
                     eval_strategy="epoch", save_strategy="epoch", fp16=True, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=data["train"],
                         eval_dataset=data["validation"], peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    print(f"Adapter saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
