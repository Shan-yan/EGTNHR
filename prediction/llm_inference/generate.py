"""Run local Hugging Face inference over a KARE JSONL test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, dtype: str):
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    path = Path(model_path).expanduser()
    kwargs = {"device_map": "auto", "torch_dtype": torch_dtype, "attn_implementation": "eager"}
    if path.is_dir() and (path / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full file")
    args = parser.parse_args(argv)

    model, tokenizer = load_model(args.model, args.dtype)
    records = [json.loads(line) for line in args.test_file.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    results = {}
    device = next(model.parameters()).device
    for index, sample in enumerate(tqdm(records, desc="Generating")):
        messages = [{"role": "user", "content": sample["input"]}]
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = sample["input"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        results[str(index)] = {
            "input": sample["input"],
            "ground_truth": sample.get("output"),
            "reasoning_and_prediction": tokenizer.decode(generated, skip_special_tokens=True),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output} ({len(results)} predictions)")


if __name__ == "__main__":
    main()
