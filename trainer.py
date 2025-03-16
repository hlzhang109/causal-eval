
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# z1 - BBH, z2 - IFEval, z3 - MATH.
# IFEval BBH MATH LVl 5 GPQA MUSR MMLU-PRO

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

sft_config = SFTConfig(output_dir="/tmp")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()