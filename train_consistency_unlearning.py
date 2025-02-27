import random
from datasets import concatenate_datasets
import datasets
from utils.utils import get_dataset, get_intermediate_outputs
from transformers import CLIPTextModel
import argparse
import logging
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.amp import autocast
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from typing import Any, List
from torch.utils.data import Sampler
from copy import deepcopy

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    step,
    torch_dtype,
    seed=None,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images_per_prompt} images with prompt:"
        f" {args.validation_prompts}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(seed) if seed else None

    prompts = pipeline_args["prompt"]
    num_images_per_prompt = pipeline_args["num_images_per_prompt"]

    images = []
    for prompt in prompts:
        with autocast('cuda'):
            with torch.no_grad():
                imgs = pipeline(prompt=prompt, num_images_per_prompt=num_images_per_prompt, generator=generator).images
                images.extend(imgs)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            if is_wandb_available():
                import wandb
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{index}: {prompt}")
                        for index, (prompt, start_idx) in enumerate(zip(prompts, range(0, len(images), num_images_per_prompt)))
                        for image in images[start_idx:start_idx + num_images_per_prompt]
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def arg_to_list(separator: str):
    def str2list(s):
        return [item.strip() for item in s.split(separator) if item.strip()]
    return str2list


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Consistency Unlearning training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--unlearn_data_dirs",
        type=arg_to_list(';'),
        default=None,
        required=True,
        help="Folders containing the unlearning training data, comma-separated",
    )
    parser.add_argument(
        "--unlearn_concepts",
        type=arg_to_list(';'),
        default=None,
        required=True,
        help="Concepts to unlearn, comma-separated",
    )
    parser.add_argument(
        "--unlearn_batch_size",
        type=int,
        default=1,
        help="Unlearn samples in a batch.",
    )
    parser.add_argument(
        "--retain_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the retained training data.",
    )
    parser.add_argument(
        "--retain_batch_size",
        type=int,
        default=3,
        help="Retain samples in a batch.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="The alpha parameter for the intermediate loss between unlearn and retain.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="The beta parameter for the ground truth loss between unlearn and retain.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="The gamma parameter for the ground truth loss and intermediate loss.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=arg_to_list(';'),
        default=None,
        help="Prompts that are used during validation to verify that the model is learning, semicolon-separated",
    )
    parser.add_argument(
        "--num_validation_images_per_prompt",
        type=int,
        default=6,
        help="Number of images that should be generated during validation with `validation_prompts`.",
    )
    parser.add_argument(
        "--unlearn_data_copy_multiple",
        type=int,
        default=1,
        help="Unlearn data copy multiple.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompts` multiple times: `args.num_validation_images_per_prompt`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="consistency-unlearning-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help=("wandb run name."),
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        help=("wandb mode."),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.train_batch_size = args.unlearn_batch_size + args.retain_batch_size
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args


def get_local_hf_dataset(
    unlearn_data_dirs: List[str],
    retain_data_dir: str,
    unlearn_concepts: List[str],
    retain_to_unlearn_ratio: float,  # 保留数据集是遗忘数据集的多少倍
    unlearn_data_copy_multiple=1,  # 所有遗忘数据集复制多少次
) -> datasets.Dataset:
    unlearn_dataset = concatenate_datasets([get_dataset(unlearn_data_dir) for unlearn_data_dir in unlearn_data_dirs] * unlearn_data_copy_multiple)
    retain_dataset = get_dataset(retain_data_dir, max_num_images=math.ceil(len(unlearn_dataset) * retain_to_unlearn_ratio),
                                 unlearn_concepts=unlearn_concepts, complement=True)

    unlearn_dataset = unlearn_dataset.map(lambda x: {"unlearn": True})
    retain_dataset = retain_dataset.map(lambda x: {"unlearn": False})

    dataset = concatenate_datasets([
        unlearn_dataset,
        retain_dataset
    ]).shuffle()

    return dataset


class MyDataset(Dataset):
    """
    A dataset to prepare the unlearning and retained images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.dataset = hf_dataset

        self._length = len(self.dataset)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        data = self.dataset[index]
        source_image = data['source_image']
        source_prompt = data['source_prompt']
        target_image = data['target_image']
        target_prompt = data['target_prompt']
        example['unlearn'] = data['unlearn']

        if not source_image.mode == "RGB":
            source_image = source_image.convert("RGB")
        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")

        example["source_image"] = self.image_transforms(source_image)
        example["target_image"] = self.image_transforms(target_image)

        source_text_inputs = tokenize_prompt(self.tokenizer, source_prompt)
        example["source_prompt_ids"] = source_text_inputs.input_ids
        example["source_attention_mask"] = source_text_inputs.attention_mask

        target_text_inputs = tokenize_prompt(self.tokenizer, target_prompt)
        example["target_prompt_ids"] = target_text_inputs.input_ids
        example["target_attention_mask"] = target_text_inputs.attention_mask

        return example


class UnlearnRetainBatchSampler(Sampler):
    def __init__(self, dataset: datasets.Dataset, unlearn_per_batch: int, retain_per_batch: int):
        self.dataset = dataset
        self.batch_size = unlearn_per_batch + retain_per_batch

        self.unlearn_indices = []
        self.retain_indices = []
        for i, unlearn in enumerate(dataset['unlearn']):
            if unlearn:
                self.unlearn_indices.append(i)
            else:
                self.retain_indices.append(i)
        self.unlearn_indices = self.unlearn_indices[:len(self.unlearn_indices) - len(self.unlearn_indices) % unlearn_per_batch]
        self.retain_indices = self.retain_indices[:len(self.retain_indices) - len(self.retain_indices) % retain_per_batch]
        num_batches = len(self.unlearn_indices) // unlearn_per_batch

        self.unlearn_per_batch = unlearn_per_batch
        self.retain_per_batch = retain_per_batch

        self.num_batches = num_batches

    def __iter__(self):
        unlearn_indices = self.unlearn_indices
        retain_indices = self.retain_indices
        random.shuffle(unlearn_indices)
        random.shuffle(retain_indices)

        grouped_unlearn_indices = self.group_elements(unlearn_indices, self.unlearn_per_batch)
        grouped_retain_indices = self.group_elements(retain_indices, self.retain_per_batch)

        indices = [item for pair in zip(grouped_unlearn_indices, grouped_retain_indices) for item in pair]
        indices = sum(indices, [])

        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return self.num_batches

    def group_elements(self, lst: List[Any], group_size: int) -> List[List[Any]]:
        return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]


def tokenize_prompt(tokenizer, prompt):
    max_length = tokenizer.model_max_length
    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs


def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=None,
        return_dict=False,
    )
    return prompt_embeds[0]


def set_wandb_env(args):
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_RUN_ID"] = args.wandb_run_name
    os.environ["WANDB_NAME"] = args.wandb_run_name

def train_main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    set_wandb_env(args)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    origin_unet = deepcopy(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    origin_unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    origin_unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    unet_target_modules = [
        # "proj_in", "proj_out", "conv1", "conv2", "conv", "conv_out",
        "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"
    ]
    # blocks = [
    #     "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
    #     "mid_block",
    #     "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
    # ]
    # modules = [
    #     "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    #     "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    # ]
    # unet_target_modules = [name for name, _ in unet.named_modules() if any(name.startswith(b) for b in blocks) and any(name.endswith(m) for m in modules)]
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights=True,
        target_modules=unet_target_modules,
    )
    unet.add_adapter(unet_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights=True,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make sure the trainable params are in float32.
    def ensure_fp32_params():
        if args.mixed_precision == "fp16":
            models = [unet]
            if args.train_text_encoder:
                models.append(text_encoder)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    ensure_fp32_params()

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    hf_dataset = get_local_hf_dataset(
        unlearn_data_dirs=args.unlearn_data_dirs,
        retain_data_dir=args.retain_data_dir,
        unlearn_concepts=args.unlearn_concepts,
        retain_to_unlearn_ratio=args.retain_batch_size / args.unlearn_batch_size,
        unlearn_data_copy_multiple=args.unlearn_data_copy_multiple,
    )

    # Dataset and DataLoaders creation:
    train_dataset = MyDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
    )

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=True,
    #     num_workers=args.dataloader_num_workers,
    # )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=UnlearnRetainBatchSampler(hf_dataset, unlearn_per_batch=args.unlearn_batch_size, retain_per_batch=args.retain_batch_size),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, origin_unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, origin_unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, origin_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, origin_unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed after accelerator.prepare
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if override_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(deepcopy(args))
        accelerator.init_trackers("consistency-unlearning", config=tracker_config)

    def get_timestep_weight(timesteps: torch.Tensor, timestep_weights: List[float], max_timestep=999):
        n = len(timestep_weights)
        interval_width = (max_timestep + 1) / n  # 加1是为了包含999
        indices = torch.clamp((timesteps / interval_width).long(), 0, n - 1).to(timesteps.device)
        weights = torch.tensor(timestep_weights).to(timesteps.device)
        return weights[indices]

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(0, args.num_train_epochs):
        unet.train()
        origin_unet.eval()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            source_image = batch['source_image'].to(dtype=weight_dtype)
            target_image = batch['target_image'].to(dtype=weight_dtype)
            source_prompt_ids = batch['source_prompt_ids']
            target_prompt_ids = batch['target_prompt_ids']

            unlearn = batch['unlearn']
            with autocast("cuda"):
                # Convert images to latent space
                source_input = vae.encode(source_image).latent_dist.sample() * vae.config.scaling_factor
                target_input = vae.encode(target_image).latent_dist.sample() * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(source_input)
                bsz, *_ = source_input.shape
                assert bsz == args.train_batch_size
                # Sample a random timestep for each image
                max_timestep = noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_timestep, (bsz,), device=source_input.device).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_source_input = noise_scheduler.add_noise(source_input, noise, timesteps)
                noisy_target_input = noise_scheduler.add_noise(target_input, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(text_encoder, source_prompt_ids)

                target_modules = [
                    'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
                    'mid_block',
                    'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
                ]
                noise_pred, intermediate_outputs = get_intermediate_outputs(
                    model=unet,
                    inputs={"sample": noisy_target_input, "timestep": timesteps, "encoder_hidden_states": encoder_hidden_states, "return_dict": False},
                    target_modules=target_modules
                )
                with torch.no_grad():
                    target_encoder_hidden_states = encode_prompt(text_encoder, target_prompt_ids)
                    origin_target_noise_pred, origin_target_intermediate_outputs = get_intermediate_outputs(
                        model=origin_unet,
                        inputs={"sample": deepcopy(noisy_target_input), "timestep": timesteps, "encoder_hidden_states": target_encoder_hidden_states, "return_dict": False},
                        target_modules=target_modules
                    )
                    origin_unlearn_source_noise_pred, origin_unlearn_source_intermediate_outputs = get_intermediate_outputs(
                        model=origin_unet,
                        inputs={"sample": deepcopy(noisy_source_input[unlearn]), "timestep": timesteps[unlearn], "encoder_hidden_states": deepcopy(encoder_hidden_states[unlearn]), "return_dict": False},
                        target_modules=target_modules
                    )

                layer_num = len(intermediate_outputs)
                # 不同时间步使用不同权重
                # timestep_weight = get_timestep_weight(timesteps, [0.2, 0.4, 0.6, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0], max_timestep=max_timestep-1)
                timestep_weight = get_timestep_weight(timesteps, [(1 - a).sqrt().item() for a in noise_scheduler.alphas_cumprod], max_timestep=max_timestep-1)
                # timestep_weight = get_timestep_weight(timesteps, [1], max_timestep=max_timestep-1)

                # 中间层损失
                unlearn_intermediate_loss = 0.0
                retain_intermediate_loss = 0.0

                layer_weights = [0.16, 0.16, 0.17, 0.21, 0.36, 0.39, 0.46, 0.59, 1.0]
                for level, (output, origin_output, origin_unlearn_output) in enumerate(zip(intermediate_outputs, origin_target_intermediate_outputs, origin_unlearn_source_intermediate_outputs)):
                    if unlearn.sum():
                        # 不同中间层使用不同权重
                        # layer_weight = (level + 1) / layer_num
                        layer_weight = layer_weights[level]
                        unlearn_target_loss = F.mse_loss(output[unlearn], origin_output[unlearn], reduction="none")
                        assert len(output[unlearn]) == len(origin_unlearn_output)
                        unlearn_source_loss = F.mse_loss(output[unlearn], origin_unlearn_output, reduction="none")
                        contrastive_loss = unlearn_target_loss / (unlearn_target_loss + unlearn_source_loss + 1e-7)
                        contrastive_loss = layer_weight * timestep_weight[unlearn] * contrastive_loss.mean(dim=tuple(range(1, len(output.shape))))
                        unlearn_intermediate_loss += contrastive_loss.mean()
                    if (~unlearn).sum():
                        retain_intermediate_loss += F.mse_loss(output[~unlearn], origin_output[~unlearn])

                # unlearn_intermediate_loss = torch.log1p(unlearn_intermediate_loss)
                intermediate_loss = (1 / layer_num) * (args.alpha * unlearn_intermediate_loss + (1 - args.alpha) * retain_intermediate_loss)

                # ground truth loss
                unlearn_gt_loss = 0.0
                retain_gt_loss = 0.0
                if unlearn.sum():
                    unlearn_target_gt_loss = F.mse_loss(noise_pred[0][unlearn], origin_target_noise_pred[0][unlearn], reduction="none")
                    unlearn_source_gt_loss = F.mse_loss(noise_pred[0][unlearn], origin_unlearn_source_noise_pred[0], reduction="none")
                    contrastive_gt_loss = unlearn_target_gt_loss / (unlearn_target_gt_loss + unlearn_source_gt_loss + 1e-7)
                    contrastive_gt_loss = timestep_weight[unlearn] * contrastive_gt_loss.mean(dim=tuple(range(1, len(noise_pred[0].shape))))
                    unlearn_gt_loss = contrastive_gt_loss.mean()
                if (~unlearn).sum():
                    retain_gt_loss = F.mse_loss(noise_pred[0][~unlearn], origin_target_noise_pred[0][~unlearn])
                gt_loss = args.beta * unlearn_gt_loss + (1 - args.beta) * retain_gt_loss
                print(f"--> [gt_loss]: {gt_loss}, unlearn_gt_loss: {unlearn_gt_loss}, retain_gt_loss: {retain_gt_loss}")
                print(f"==> [intermediate_loss]: {intermediate_loss}, [unlearn_intermediate_loss]: {unlearn_intermediate_loss}, [retain_intermediate_loss]: {retain_intermediate_loss}")
                loss = args.gamma * gt_loss + (1 - args.gamma) * intermediate_loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        _unet = unwrap_model(unet).to(torch.float32)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(_unet))

                        if args.train_text_encoder:
                            _text_encoder = unwrap_model(text_encoder)
                            text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(_text_encoder))
                        else:
                            text_encoder_state_dict = None

                        StableDiffusionLoraLoaderMixin.save_lora_weights(
                            save_directory=os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                            unet_lora_layers=unet_lora_state_dict,
                            text_encoder_lora_layers=text_encoder_state_dict,
                        )

            logs = {
                "loss": loss.detach().item(),
                "gt_loss": gt_loss.detach().item(),
                "unlearn_gt_loss": unlearn_gt_loss.detach().item(),
                "retain_gt_loss": retain_gt_loss.detach().item(),
                "intermediate_loss": intermediate_loss.detach().item(),
                "unlearn_intermediate_loss": unlearn_intermediate_loss.detach().item(),
                "retain_intermediate_loss": retain_intermediate_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                        safety_checker=None,
                    )

                    pipeline_args = {"prompt": args.validation_prompts, "num_inference_steps": 50, "num_images_per_prompt": args.num_validation_images_per_prompt}

                    log_validation(
                        pipeline,
                        args,
                        accelerator,
                        pipeline_args,
                        epoch,
                        seed=args.seed,
                        torch_dtype=weight_dtype,
                    )
                    ensure_fp32_params()

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)

        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder = unwrap_model(text_encoder)
            text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
        else:
            text_encoder_state_dict = None

        StableDiffusionLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_state_dict,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype, safety_checker=None,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

        # run inference
        if args.validation_prompts and args.num_validation_images_per_prompt > 0:
            pipeline_args = {"prompt": args.validation_prompts, "num_inference_steps": 50, "num_images_per_prompt": args.num_validation_images_per_prompt}
            log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args,
                epoch,
                seed=None,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

    accelerator.end_training()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    train_main(args)
