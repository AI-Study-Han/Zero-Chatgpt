#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from utils.model_utils import create_critic_model
from utils.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str)
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )

    # Evaluation
    parser.add_argument("--eval_interval",
                        type=int,
                        default=0,
                        help="If > 0, perform evaluation at this interval")
    parser.add_argument("--eval_iters",
                        type=int,
                        default=100,
                        help="Maximum evaluation iterations")
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path)
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   dropout=args.dropout,
                                   zero_stage=args.zero_stage,
                                   compute_fp32_loss=args.compute_fp32_loss)

    
    print_rank_0("create_prompt_dataset前")
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, train_phase, args.seed, tokenizer,args.max_seq_len)

        # 打印train_dataset的部分内容
    

    
    data_collator = DataCollatorReward()
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)





    def evaluation_reward(model, dataloader, eval_iters):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        chosen_scores = 0.
        rejected_scores = 0.
        for _step, _batch in enumerate(dataloader):
            _batch = to_device(_batch, device)
            with torch.no_grad():
                _outputs = model(**_batch)

            chosen = _outputs["chosen_mean_scores"]
            rejected = _outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            chosen_scores += _outputs["chosen_mean_scores"].mean().float()
            rejected_scores += _outputs["rejected_mean_scores"].mean().float()
            if (_step + 1) == eval_iters:
                break
        _acc = correct_predictions / total_predictions
        chosen_scores = chosen_scores / (_step + 1)
        rejected_scores = rejected_scores / (_step + 1)
        try:
            _acc = get_all_reduce_mean(_acc).item()
            chosen_scores = get_all_reduce_mean(chosen_scores).item()
            rejected_scores = get_all_reduce_mean(rejected_scores).item()
        except:
            pass
        return chosen_scores, rejected_scores, _acc
    
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = FusedAdam

    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
 
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
 
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
  
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
   
    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, reject_score, acc = evaluation_reward(
        rm_model, eval_dataloader, args.eval_iters)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, "
        f"rejected_last_scores (lower is better) : {reject_score}, "
        f"acc (higher is better) : {acc}", args.global_rank)

    total_micro_steps = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            total_micro_steps += 1
            gas_boundary = (total_micro_steps %
                            args.gradient_accumulation_steps == 0)
            total_steps = total_micro_steps // args.gradient_accumulation_steps
            if args.eval_interval and gas_boundary and (
                    total_steps % args.eval_interval == 0):
                print_rank_0(f"Iter {total_steps}: Evaluating reward",
                             args.global_rank)
                reward_score, reject_score, acc = evaluation_reward(
                    rm_model, eval_dataloader, args.eval_iters)
                print_rank_0(
                    f"Iter {total_steps}: c_scores: {reward_score}, r_scores: {reject_score}, "
                    f"diff: {reward_score - reject_score}, acc: {acc}",
                    args.global_rank)
                rm_model.train()

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, reject_score, acc = evaluation_reward(
            rm_model, eval_dataloader, args.eval_iters)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, "
            f"rejected_last_scores (lower is better) : {reject_score}, "
            f"acc (higher is better) : {acc}", args.global_rank)
        rm_model.tput_timer.update_epoch_count()

        if args.output_dir is not None:
            print_rank_0('saving model ...', args.global_rank)

            if args.global_rank == 0:
                save_hf_format(rm_model, tokenizer, args, sub_folder=f"epoch_{epoch+1}")
            if args.zero_stage == 3:
                raise RuntimeError('不支持zero3')


if __name__ == "__main__":
    main()
