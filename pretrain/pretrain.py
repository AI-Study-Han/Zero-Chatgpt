from dataclasses import dataclass
from typing import Optional
from transformers.utils.versions import require_version
import transformers
from model.modeling_miaomiao import MiaomiaoForCausalLM
from model.configuration_miaomiao import MiaomiaoConfig
from pretrain_dataset import PretrainDataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
import torch
import json
import os
import logging
import glob
import random
import numpy as np

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
    config_file: Optional[str] = None
    torch_dtype: Optional[str] = None

@dataclass
class DataTrainingArguments:
    train_dataset_dir: Optional[str] = None
    block_size: Optional[int] = None
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    

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

def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

def init_model(training_args, model_args):
    if training_args.init_from == "scratch":
        config = MiaomiaoConfig.from_pretrained(model_args.config_file)
        print(config)
        model = MiaomiaoForCausalLM(config)
        return model
    
    

def my_data_collator(input_datas):
        # 将所有样本的输入 (`X`) 和标签 (`Y`) 分别堆叠
        input_ids = torch.stack([input_data[0] for input_data in input_datas])
        labels = torch.stack([input_data[1] for input_data in input_datas])

        # 返回一个字典，包含模型需要的键和值
        return {
            "input_ids": input_ids,
            "labels": labels
        }

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

    model=init_model(training_args, model_args)
    model.to(training_args.use_device)

    if training_args.use_compile:
        model = torch.compile(model)

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}")
    logger.info(f"可训练参数: {trainable_params}")

    logger.info(f"torch_dtype:{model_args.torch_dtype}")
    logger.info(f"training_args.bf16: {training_args.bf16}")
    

    train_data_path_list = glob.glob(os.path.join(data_args.train_dataset_dir, '*.bin'))
    train_ds = PretrainDataset(train_data_path_list, max_length=data_args.block_size, memmap=True, seed=training_args.seed)
    logger.info(f"Train dataset size: {len(train_ds)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=my_data_collator,
        callbacks=[LoggingCallback(logger)],  # 添加自定义回调
    )
    print(training_args.bf16)

    trainer.train()




if __name__ == "__main__":
    main()
