import shutil
from diffusers import StableDiffusionPipeline
import torch
from typing import List,  Tuple
from pathlib import Path
from utils.utils import read_file_lines, duplicate_prompts

negative_prompt = "low quality, blurry, zoom in, zoom out, anime"
device = "cuda" if torch.cuda.is_available() else "cpu"

sd_pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.bfloat16, safety_checker=None).to(device)
sd_pipeline.enable_attention_slicing()
sd_pipeline.fuse_qkv_projections()

guidance_scale = 7.5


def generate_images(prompts: List[str], out_dir: str, batch_size: int = 4, target_size: Tuple[int] = (512, 512),
                    num_inference_steps: int = 50, guidance_scale: float = guidance_scale):
    path = Path(out_dir)
    if path.exists():
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        images = sd_pipeline(batch_prompts,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale,
                             negative_prompt=[negative_prompt]*len(batch_prompts)).images
        for j, img in enumerate(images):
            index = i + j + 1
            file_name = f"{index:04}.png"
            file_path = f"{out_dir}/{file_name}"
            image = img.convert("RGB").resize(target_size)
            image.save(file_path)

        torch.cuda.empty_cache()


def generate_unlearn_target_images(unlearn_concept: str, target_concept: str, data_root="./data", image_num=500, batch_size=50):
    data_dir = f"{data_root}/_{unlearn_concept}"

    unlearn_dir = f"{data_dir}/_unlearn"  # 要遗忘的图片
    target_dir = f"{data_dir}/_target"  # 要遗忘的图片

    Path(unlearn_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    unlearn_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{unlearn_concept}.txt")
    target_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{target_concept}.txt")

    unlearn_prompts, target_prompts = duplicate_prompts(image_num, unlearn_prompts, target_prompts)

    generate_images(unlearn_prompts, out_dir=unlearn_dir, batch_size=batch_size)
    generate_images(target_prompts, out_dir=target_dir, batch_size=batch_size)


def re_generate_unlearn_target_image(unlearn_concept: str, target_concept: str, seqs: List[int], data_root="./data", image_num=500, generate_unlearn=True, generate_target=True):
    data_dir = f"{data_root}/_{unlearn_concept}"

    unlearn_dir = f"{data_dir}/_unlearn"
    target_dir = f"{data_dir}/_target"

    assert Path(unlearn_dir).exists()
    assert Path(target_dir).exists()

    unlearn_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{unlearn_concept}.txt")
    target_prompts = read_file_lines(f"./prompts/{unlearn_concept}/{target_concept}.txt")

    unlearn_prompts, target_prompts = duplicate_prompts(image_num, unlearn_prompts, target_prompts)

    for seq in seqs:
        if generate_unlearn:
            generate_image(unlearn_prompts[seq-1], save_path=f"{unlearn_dir}/{seq:04}.png")
        if generate_target:
            generate_image(target_prompts[seq-1], save_path=f"{target_dir}/{seq:04}.png")


def generate_image(prompt: str, save_path: str, target_size: Tuple[int] = (512, 512), num_inference_steps: int = 50,
                   guidance_scale: float = guidance_scale, negative_prompt: str = negative_prompt):
    img = sd_pipeline(prompt, num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale,
                      negative_prompt=negative_prompt).images[0]
    image = img.convert("RGB").resize(target_size)
    image.save(save_path)
