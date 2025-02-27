# import os
# if True:
#     os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import shutil
from diffusers import StableDiffusionXLPipeline
import torch
from typing import List,  Tuple
from pathlib import Path
from utils.utils import read_file_lines
import random

negative_prompt = "low quality, blurry, zoom in, zoom out, anime"
device = "cuda" if torch.cuda.is_available() else "cpu"

sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, safety_checker=None).to(device)
sdxl_pipeline.enable_attention_slicing()
sdxl_pipeline.fuse_qkv_projections()

guidance_scale = 5


def generate_images(prompts: List[str], out_dir: str, batch_size: int = 4, target_size: Tuple[int] = (512, 512),
                    num_inference_steps: int = 50, guidance_scale: float = guidance_scale):
    path = Path(out_dir)
    if path.exists():
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        images = sdxl_pipeline(batch_prompts,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               negative_prompt=[negative_prompt]*len(batch_prompts)).images
        for j, img in enumerate(images):
            index = i + j + 1
            file_name = f"{index:04}.png"
            file_path = f"{out_dir}/{file_name}"
            image = img.convert("RGB").resize(target_size)
            image.save(file_path)


def generate_unlearn_target_images(unlearn_concept: str, target_concept: str, data_root="./data", image_num=500):
    data_dir = f"{data_root}/_{unlearn_concept}"

    unlearn_dir = f"{data_dir}/_unlearn"  # 要遗忘的图片
    target_dir = f"{data_dir}/_target"  # 要遗忘的图片

    Path(unlearn_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    unlearn_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{unlearn_concept}.txt")
    target_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{target_concept}.txt")
    assert len(unlearn_prompts) == len(target_prompts)

    pair_prompts = list(zip(unlearn_prompts, target_prompts))
    random.shuffle(pair_prompts)
    copied_pair_prompts = pair_prompts * (image_num // len(unlearn_prompts))
    if image_num % len(unlearn_prompts) != 0:
        copied_pair_prompts += pair_prompts[:image_num % len(unlearn_prompts)]
    print("copied_pair_prompts", copied_pair_prompts)
    unlearn_prompts = [pair[0] for pair in copied_pair_prompts]
    target_prompts = [pair[1] for pair in copied_pair_prompts]
    generate_images(unlearn_prompts, out_dir=unlearn_dir, batch_size=6)
    generate_images(target_prompts, out_dir=target_dir, batch_size=6)


def re_generate_unlearn_target_image(unlearn_concept: str, target_concept: str, seqs: List[int], data_root="./data"):
    data_dir = f"{data_root}/_{unlearn_concept}"

    unlearn_dir = f"{data_dir}/_unlearn"
    target_dir = f"{data_dir}/_target"

    assert Path(unlearn_dir).exists()
    assert Path(target_dir).exists()

    unlearn_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{unlearn_concept}.txt")
    target_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{target_concept}.txt")

    for seq in seqs:
        generate_image(unlearn_prompts[seq-1], save_path=f"{unlearn_dir}/{seq:04}.png")
        generate_image(target_prompts[seq-1], save_path=f"{target_dir}/{seq:04}.png")


def generate_image(prompt: str, save_path: str, target_size: Tuple[int] = (512, 512), num_inference_steps: int = 50,
                   guidance_scale: float = guidance_scale, negative_prompt: str = negative_prompt):
    img = sdxl_pipeline(prompt, num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt).images[0]
    image = img.convert("RGB").resize(target_size)
    image.save(save_path)
