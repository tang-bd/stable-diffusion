import argparse, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
from einops import repeat, rearrange
from kornia.augmentation import RandomAffine, RandomPerspective, RandomGaussianNoise
import torch
from torchvision.transforms.v2.functional import normalize
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def preprocess(x_in, num_augmentations=1):
        x_in = torch.nn.functional.interpolate(
            x_in, size=(336, 336), mode="bilinear"
        )
        x_in = repeat(x_in, "1 c h w -> b c h w", b=num_augmentations)
        if num_augmentations > 1:
            augmentations = torch.nn.Sequential(
                RandomGaussianNoise(0.1, p=0.5),
                RandomAffine(degrees=10, translate=0.1, p=0.5, padding_mode="border"),
                RandomPerspective(0.1, p=0.5),
            )
            x_in = augmentations(x_in)
        x_in = ((x_in + 1) / 2).clamp(0, 1)
        x_in = normalize(
            x_in,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        return x_in


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["a painting of a virus monster playing guitar"],
        help="the prompts to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--mllm_path",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        nargs='+',
        default=["Describe this image in detail."],
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cg_scale",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--gradient_noise",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stopping_step",
        type=int,
        default=None,
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    if not opt.from_file:
        if len(opt.instructions) == 1:
            all_instructions = [opt.instructions] * len(opt.prompts)
            all_prompts = [opt.prompts]
        elif len(opt.instructions) != len(opt.prompts):
            raise ValueError("instructions and prompt must be of the same length")
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            all_instructions = [opt.instructions] * len(data)
            all_prompts = [[prompt] for prompt in list(data)]

    if opt.stopping_step is None:
        opt.stopping_step = opt.ddim_steps

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    n_rows = opt.n_rows if opt.n_rows > 0 else 1

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    tokenizer, mllm, _, max_length = load_pretrained_model(
        model_path=opt.mllm_path,
        model_base=opt.model_base,
        model_name=get_model_name_from_path(opt.mllm_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    mllm.eval()

    all_input_ids_list = []
    all_labels_list = []
    all_attention_mask_list = []
    for instructions, prompts in zip(all_instructions, all_prompts):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        for instruction, prompt in zip(instructions, prompts):
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{instruction}")
            conv.append_message(conv.roles[1], None)
            instruction = tokenizer_image_token(
                conv.get_prompt(),
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )[:max_length]
            conv.messages[-1][1] = prompt
            full = tokenizer_image_token(
                conv.get_prompt(),
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )[:max_length]

            input_ids = full.unsqueeze(0)
            labels = full.clone()
            labels[: len(instruction)] = IGNORE_INDEX
            labels = labels.unsqueeze(0)
            attention_mask = full.ne(IGNORE_INDEX).unsqueeze(0)
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
        all_input_ids_list.append(input_ids_list)
        all_labels_list.append(labels_list)
        all_attention_mask_list.append(attention_mask_list)

    with precision_scope("cuda"):
        with model.ema_scope():
            tic = time.time()
            for prompts, input_ids_list, labels_list, attention_mask_list in tqdm(zip(all_prompts, all_input_ids_list, all_labels_list, all_attention_mask_list), total=len(all_prompts)):
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    uc = None
                    if opt.cfg_scale != 1.0:
                        uc = model.get_learned_conditioning([""])
                    c = model.get_learned_conditioning([prompts[0]])

                    def cond_fn(x):
                        x = model.decode_first_stage(x)
                        x = preprocess(x, opt.num_augmentations)
                        total_loss = torch.tensor(0.0, device=device)
                        for input_ids, labels, attention_mask in zip(input_ids_list, labels_list, attention_mask_list):
                            for i in range(x.shape[0]):
                                images = x[i].unsqueeze(0)
                                output = mllm.forward(
                                    input_ids=input_ids.to(device=device),
                                    labels=labels.to(device=device),
                                    attention_mask=attention_mask.to(device=device),
                                    images=images.to(dtype=mllm.dtype),
                                )

                                total_loss += output.loss
                        
                        return -opt.cg_scale * total_loss / opt.num_augmentations / len(input_ids_list)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.cfg_scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        cond_fn=cond_fn if opt.cg_scale > 0.0 else None,
                                                        gradient_noise=opt.gradient_noise,
                                                        stopping_step=opt.stopping_step,
                                                        )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    if not opt.skip_save:
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{prompts[0]}_{base_count:06}.png"))
                            base_count += 1

                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'{prompts[0]}-grid-{grid_count:04}.png'))
                    grid_count += 1

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
