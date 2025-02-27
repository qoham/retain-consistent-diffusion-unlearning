import random
from torch import nn
from torch.utils.hooks import RemovableHandle
import torch
from typing import Any, Dict, Union, List
from datasets import Features, Value, Image, Dataset
import shutil
import json
from pathlib import Path


def write_jsonl(data: List[Dict], file_path: str = "metadata.jsonl"):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def rm_and_mkdir(dir: str):
    path = Path(dir)
    if path.exists():
        shutil.rmtree(dir)
    Path(dir).mkdir(parents=True, exist_ok=True)


def read_file_lines(file_path: Path) -> List[str]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data.append(line.strip())
    return data


def copy_directory(src_dir: str, dst_dir: str, prefix: str = ""):
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        new_name = f"{prefix}{item.name}"
        target_path = dst / new_name

        if item.is_file():
            shutil.copy2(item, target_path)
        elif item.is_dir():
            copy_directory(item, target_path, prefix)


def generate_jsonl(out_dir: str, source_prompts: List[str], target_prompts: List[str], source_img_dir: str, target_img_dir: str):
    assert Path(out_dir).exists(), f"Output directory {out_dir} does not exist"

    copy_directory(source_img_dir, out_dir, prefix="source_image_")
    copy_directory(target_img_dir, out_dir, prefix="target_image_")

    data = [
        {"source_image": f"{out_dir}/source_image_{(i+1):04}.png", "source_prompt": source_prompt,
         "target_image": f"{out_dir}/target_image_{(i+1):04}.png", "target_prompt": target_prompt}
        for i, (source_prompt, target_prompt) in enumerate(zip(source_prompts, target_prompts))
    ]
    write_jsonl(data, f"{out_dir}/metadata.jsonl")


def generate_metadata_jsonl(out_dir: str, source_prompts: List[str], target_prompts: List[str]):
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = [{"source_prompt": source_prompt, "target_prompt": target_prompt}
            for source_prompt, target_prompt in zip(source_prompts, target_prompts)]
    write_jsonl(data, f"{out_dir}/metadata.jsonl")


def get_metadata_dataset(data_dir: str):
    metadata = read_jsonl(f"{data_dir}/metadata.jsonl")
    dataset = Dataset.from_list(metadata)
    features = Features({
        "source_prompt": Value("string"),
        "target_prompt": Value("string"),
    })
    dataset = dataset.cast(features)
    return dataset


def get_dataset(data_dir: str, max_num_images: int = None, unlearn_concepts: List[str] = None, complement=True):
    metadata = read_jsonl(f"{data_dir}/metadata.jsonl")
    if unlearn_concepts:
        metadata = [m for m in metadata if all([c.lower() not in m['source_prompt'].lower() for c in unlearn_concepts])]
    if max_num_images is None:
        max_num_images = 0
    metadata = metadata[-max_num_images:]
    if len(metadata) < max_num_images and complement:
        metadata = metadata + random.choices(metadata, k=max_num_images - len(metadata))
    dataset = Dataset.from_list(metadata)
    features = Features({
        "source_image": Image(),
        "source_prompt": Value("string"),
        "target_image": Image(),
        "target_prompt": Value("string"),
    })
    dataset = dataset.cast(features)
    return dataset


def dict_to_argv(d):
    argv = []
    for key, value in d.items():
        if isinstance(value, bool):
            if value:
                argv.append(f'--{key}')
        else:
            argv.append(f'--{key}')
            argv.append(str(value))
    return argv


def get_intermediate_outputs(model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], target_modules: List[str]):
    intermediate_outputs = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        intermediate_outputs.append(output)

    hooks: List[RemovableHandle] = []
    for name, module in model.named_modules():
        if name in target_modules or any(name.endswith(module) for module in target_modules):
            hook_handle = module.register_forward_hook(hook)
            hooks.append(hook_handle)

    outputs = model(**inputs)

    for h in hooks:
        h.remove()

    return outputs, intermediate_outputs


def freeze_model(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False


def unfreeze_lora_module(model: nn.Module):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


def create_unlearn_dataset(unlearn_concept: str, target_concept: str, image_num=500):
    img_dir = f"./data/_{unlearn_concept}"

    unlearn_img_dir = f"{img_dir}/_unlearn"  # 要遗忘的图片
    target_img_dir = f"{img_dir}/_target"  # 跟遗忘图片对应的目标图片

    assert Path(unlearn_img_dir).exists()
    assert Path(target_img_dir).exists()

    data_dir = f"./data/{unlearn_concept}"

    unlearn_path = Path(data_dir)
    if unlearn_path.exists():
        shutil.rmtree(data_dir)
    unlearn_path.mkdir(parents=True, exist_ok=True)

    unlearn_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{unlearn_concept}.txt")
    target_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{target_concept}.txt")
    
    unlearn_prompts, target_prompts = duplicate_prompts(image_num, unlearn_prompts, target_prompts)

    generate_jsonl(
        out_dir=data_dir,
        source_prompts=unlearn_prompts, target_prompts=target_prompts,
        source_img_dir=unlearn_img_dir, target_img_dir=target_img_dir,
    )

def duplicate_prompts(image_num: int, unlearn_prompts: List[str], target_prompts: List[str]):
    assert len(unlearn_prompts) == len(target_prompts)
    
    pair_prompts = list(zip(unlearn_prompts, target_prompts))
    copied_pair_prompts = pair_prompts * (image_num // len(unlearn_prompts))
    if image_num % len(unlearn_prompts) != 0:
        copied_pair_prompts += pair_prompts[:image_num % len(unlearn_prompts)]
    unlearn_prompts = [pair[0] for pair in copied_pair_prompts]
    target_prompts = [pair[1] for pair in copied_pair_prompts]
    return unlearn_prompts,target_prompts


def create_other_dataset():
    other_img_dir = f"./data/_other"

    assert Path(other_img_dir).exists()

    other_dir = f"./data/other"

    other_path = Path(other_dir)
    if other_path.exists():
        shutil.rmtree(other_dir)
    other_path.mkdir(parents=True, exist_ok=True)

    other_prompts = read_file_lines(f"./prompts/other/other.txt")

    generate_jsonl(
        out_dir=other_dir,
        source_prompts=other_prompts, target_prompts=other_prompts,
        source_img_dir=other_img_dir, target_img_dir=other_img_dir,
    )
