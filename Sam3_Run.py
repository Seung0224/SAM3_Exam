# SAM3 패키지 버전
# Fine Tuning 진행 시 transformers 버전으로 변환 필요

import os
from typing import List, Optional

import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = r"D:\1.png"
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

# 한 개만 쓸 때도 ["dark blob"] 처럼 리스트로 쓰면 됨
# "dark blob", "circle", "bright spot", "etc"
TEXT_PROMPTS: List[str] = [
    "dark blob"
]

# 한 프롬프트에서 여러 인스턴스가 나왔을 때 처리 방식
# True  : 해당 프롬프트마다 점수가 가장 높은 마스크 하나만 사용
# False : 해당 프롬프트에서 나온 모든 마스크를 합쳐서 사용(OR)
USE_ONLY_BEST_MASK_PER_PROMPT = False

MASK_PROB_THRESHOLD = 0.3     # 그대로 유지
MASK_ALPHA = 0.25             # 더 투명하게 (0.0~1.0)

PROMPT_COLORS_RGB = [
    (0, 128, 255),   # 파란색 계열 (첫 번째 프롬프트)
    (0, 255, 255),   # 시안 (두 번째 프롬프트)
    (255, 0, 255),   # 마젠타
    (0, 255, 0),     # 초록
]


# =========================================
# SAM3 + 오버레이 유틸 함수
# =========================================

def load_sam3() -> Sam3Processor:
    print("[INFO] Loading SAM3 model ...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("[INFO] SAM3 model + processor loaded.")
    return processor

def run_sam3_text_prompts(processor: Sam3Processor, image_path: str, text_prompts: List[str]):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 이미지 로드 (RGB)
    image = Image.open(image_path).convert("RGB")

    # SAM3 상태 한 번만 설정
    inference_state = processor.set_image(image)

    results = []

    for prompt in text_prompts:
        print(f"[INFO] Running text prompt: '{prompt}'")
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

        masks = output["masks"].detach().cpu()
        boxes = output["boxes"].detach().cpu()
        scores = output["scores"].detach().cpu()

        print(f"       -> Detected {masks.shape[0]} mask(s) for this prompt.")

        results.append(
            {
                "prompt": prompt,
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
            }
        )

    return image, results

def _masks_to_binary_hw(masks_tensor, scores_tensor, use_only_best: bool, prob_threshold: float) -> Optional[np.ndarray]:
    """
    SAM3가 반환한 masks / scores에서 최종 사용할 바이너리 마스크 생성.
    반환: [H, W] (0/1) 또는 None
    """
    # masks: [N, H, W] 또는 [N, 1, H, W]
    if masks_tensor.ndim == 4:
        # [N, 1, H, W] -> [N, H, W]
        masks_np = masks_tensor[:, 0, :, :].numpy()
    elif masks_tensor.ndim == 3:
        # [N, H, W]
        masks_np = masks_tensor.numpy()
    else:
        raise ValueError(f"Unexpected masks shape: {masks_tensor.shape}")

    num_masks = masks_np.shape[0]
    if num_masks == 0:
        return None

    if use_only_best:
        best_idx = int(scores_tensor.argmax().item())
        selected = masks_np[best_idx]  # [H, W]
        binary = (selected > prob_threshold).astype(np.uint8)
    else:
        # 모든 마스크를 OR로 합침
        binary = (masks_np > prob_threshold).any(axis=0).astype(np.uint8)

    if binary.sum() == 0:
        return None

    return binary  # [H, W], 0 또는 1

def overlay_one_mask(base_rgba: Image.Image,mask_hw: np.ndarray,color_rgb,alpha: float,) -> Image.Image:
    if mask_hw is None:
        return base_rgba

    base = base_rgba.convert("RGBA")

    w, h = base.size  # size = (W,H)
    if mask_hw.shape != (h, w):
        mask_img = Image.fromarray(mask_hw.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize(base.size, resample=Image.NEAREST)
    else:
        mask_img = Image.fromarray(mask_hw.astype(np.uint8) * 255, mode="L")

    r, g, b = color_rgb
    overlay = Image.new("RGBA", base.size, (r, g, b, 0))

    def _scale(v):
        return int(v * float(alpha))

    alpha_mask = mask_img.point(_scale)
    overlay.putalpha(alpha_mask)

    result = Image.alpha_composite(base, overlay)
    return result

def save_overlay_image(overlay_image: Image.Image,image_path: str,output_dir: str,) -> str:
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    out_name = f"{name}_sam3_overlay.png"
    out_path = os.path.join(output_dir, out_name)

    overlay_image.save(out_path, format="PNG")
    return out_path

# =========================================
# 메인 실행부
# =========================================

def main():
    print(f"[INFO] Input image : {IMAGE_PATH}")
    print(f"[INFO] Output dir  : {OUTPUT_DIR}")
    print(f"[INFO] Text prompts ({len(TEXT_PROMPTS)}):")
    for i, p in enumerate(TEXT_PROMPTS, 1):
        print(f"       {i}. {p}")

    processor = load_sam3()

    image_pil, results = run_sam3_text_prompts(
        processor=processor,
        image_path=IMAGE_PATH,
        text_prompts=TEXT_PROMPTS,
    )

    # 최종 오버레이용 베이스 (RGBA)
    overlay_rgba = image_pil.convert("RGBA")

    any_mask_used = False

    for idx, res in enumerate(results):
        prompt = res["prompt"]
        masks = res["masks"]
        scores = res["scores"]
        
        print(prompt, masks.shape, scores) 

        color = PROMPT_COLORS_RGB[idx % len(PROMPT_COLORS_RGB)]

        mask_hw = _masks_to_binary_hw(
            masks_tensor=masks,
            scores_tensor=scores,
            use_only_best=USE_ONLY_BEST_MASK_PER_PROMPT,
            prob_threshold=MASK_PROB_THRESHOLD,
        )

        if mask_hw is None:
            print(f"[WARN] Prompt '{prompt}' 에 대해 유효한 마스크가 없습니다.")
            continue

        print(f"[INFO] Prompt '{prompt}' -> overlay color {color}, alpha={MASK_ALPHA}")
        overlay_rgba = overlay_one_mask(
            base_rgba=overlay_rgba,
            mask_hw=mask_hw,
            color_rgb=color,
            alpha=MASK_ALPHA,
        )
        any_mask_used = True

    if not any_mask_used:
        print("[WARN] 어떤 프롬프트에서도 마스크를 생성하지 못했습니다.")
        return

    out_path = save_overlay_image(
        overlay_image=overlay_rgba,
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR,
    )

    print(f"[OK] Saved overlay image to:\n     {out_path}")


if __name__ == "__main__":
    main()
