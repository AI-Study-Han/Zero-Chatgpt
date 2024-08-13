from dataclasses import dataclass
from typing import Optional
import transformers
from sft_dataset import SFTDataset
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
import torch
import os
import logging
import glob
import random
import numpy as np
from typing import Dict, Optional, Sequence

IGNORE_INDEX = -100
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logger.info(logs)


@dataclass
class ModelArguments:
    model_path: Optional[str] = None
    torch_dtype: Optional[str] = None

@dataclass
class DataTrainingArguments:
    train_dataset_file: Optional[str] = None
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    block_size: Optional[int] = None
    

@dataclass
class MyTrainingArguments(TrainingArguments):
    modules_to_save: Optional[str] = None

    
    # 模型初始化方式
    init_from: Optional[str] = "scratch"
    use_device: Optional[str] = 'cuda'
    use_compile: Optional[bool] = False
    log_file: Optional[str] = None
    nnodes: Optional[int] = None
    nproc_per_node: Optional[int] = None


def init_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        trust_remote_code=True
    )
    return tokenizer, model
    
    

@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        print(f"DataCollatorForSFTDataset:{input_ids}")
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志记录器
    logging.basicConfig(filename=training_args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # 创建文件处理器，并设置写模式
    file_handler = logging.FileHandler(training_args.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # 输出日志到控制台（可选）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    set_seed(training_args.seed)

    tokenizer, model =init_model(model_args)
    model.to(training_args.use_device)

    if training_args.use_compile:
        model = torch.compile(model)

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}")
    logger.info(f"可训练参数: {trainable_params}")

    logger.info(f"torch_dtype:{model_args.torch_dtype}")
    logger.info(f"training_args.bf16: {training_args.bf16}")
    

    train_ds = SFTDataset(data_path=data_args.train_dataset_file, tokenizer=tokenizer, max_length=data_args.block_size, prompt_max_len=int(data_args.block_size/2), answer_max_len=int(data_args.block_size/2), seed=training_args.seed)
    logger.info(f"Train dataset size: {len(train_ds)}")


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        callbacks=[LoggingCallback(logger)],  # 添加自定义回调
    )
    print(training_args.bf16)

    trainer.train()




if __name__ == "__main__":
    main()
