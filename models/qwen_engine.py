import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training
)

class Qwen2_5_VLEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def load_model_for_training(self):
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=True
        )
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print("Loading model (4-bit)...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.config.use_cache = False

        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing
        )
        
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        print("Injecting LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=list(self.config.lora.target_modules),
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

        return self.processor, self.model