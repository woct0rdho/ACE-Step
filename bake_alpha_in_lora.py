#!/usr/bin/env python3

import argparse
import json
from math import sqrt

import safetensors.torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--lora_config_path", type=str)
    args = parser.parse_args()

    with open(args.lora_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    rank = config["r"]
    alpha = config["lora_alpha"]
    if config["use_rslora"]:
        alpha /= sqrt(rank)
    else:
        alpha /= rank
    sqrt_alpha = sqrt(alpha)
    print("sqrt_alpha", sqrt_alpha)

    tensors = safetensors.torch.load_file(args.input_name)
    for k in tensors:
        if k.endswith((".lora_A.weight", ".lora_B.weight")):
            tensors[k] *= sqrt_alpha
    safetensors.torch.save_file(tensors, args.output_name)


if __name__ == "__main__":
    main()
