#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import torch

from utils.model_utils import create_critic_model
from utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
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
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path)
    model = create_critic_model(model_name_or_path,
                                tokenizer,
                                None,
                                num_padding_at_beginning,
                                dropout=0)

    return model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token=None):
    chosen_sentence = prompt + good_ans
    reject_sentence = prompt + bad_ans
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token=None):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def run_pair_comparison():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = None
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     additional_special_tokens)
    rm_model.to(device)
    rm_model.eval()

    prompt_list = [
        "<|im_start|>system\n你是一个由喵阿姨开发的喵喵小助手<|im_end|>\n<|im_start|>user\n帮我生成一些音乐热评<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\n你是一个由喵阿姨开发的喵喵小助手<|im_end|>\n<|im_start|>user\n根据开头，续写古诗：\n翠幄千章荫晚空<|im_end|>\n<|im_start|>assistant\n"
    ]
    good_ans_list = [
        "1、1997年听了耀威的《有缘千里》专辑，到今年20年了，一直关注，有没有像我一样的朋友？\n2、爱的故事·上集·万屡爱意寄窗扉\n爱的故事·下集·我愿他能珍惜你\n爱的故事·曲终·只有我懂得自己<|im_end|>",
        "年华心赏两无穷。云头欲落催诗雨，池面微生解愠风。经笥使君谈似绮，仙舟令尹饮如虹。娵隅自适清池乐，不信参军是郝隆。<|im_end|>"
    ]
    bad_ans_list = [
        "1、我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，我一直都觉得，这首歌是我的最爱，<|im_end|>",
        "金蟾照影照金蟾。玉兔飞来玉兔飞，玉兔飞来玉兔飞。<|im_end|>"
    ]

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(
            prompt,
            good_ans,
            bad_ans,
            tokenizer,
            max_seq_len=512,
            end_of_conversation_token=None)
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())


def run_single_sample():
    args = parse_args()
    device = torch.device(get_accelerator().device_name())

    args.end_of_conversation_token = None
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     additional_special_tokens)
    rm_model.to(device)

    prompt = "Human: Explain the moon landing to a 6 year old in a few sentences."
    my_ans = "Assistant: The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another"

    batch = prepare_singlesample(
        prompt,
        my_ans,
        tokenizer,
        max_seq_len=512,
        end_of_conversation_token=args.end_of_conversation_token)
    batch = to_device(batch, device)

    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch, prompt_length=max(2, args.num_padding_at_beginning)
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["chosen_end_scores"].item())


if __name__ == "__main__":
    run_pair_comparison()
    # run_single_sample()
