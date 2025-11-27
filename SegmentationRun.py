"""
SAM3 패키지 버전 → YOLO 세그멘테이션용 annotation txt 자동 생성 스크립트

기능:
1) IMAGE_DIR 안의 모든 이미지 파일(.png, .jpg, .jpeg, .bmp)을 처리
2) 바탕화면에 'sam3result' 폴더 생성
3) 각 이미지마다 같은 이름의 txt 파일을 sam3result 폴더에 생성
   - 형식: class_id x1 y1 x2 y2 ... (YOLO segmentation polygon, 좌표 0~1)
   - class_id 는 CLASS_ID 상수로 제어 (기본 0)

주의:
- polygon 추출을 위해 opencv-python 필요
    pip install opencv-python
"""

import os
from typing import List, Optional

import numpy as np
from PIL import Image
import cv2  # pip install opencv-python

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =========================================
# 🔧 사용자 설정 구역
# =========================================

# 1) 처리할 이미지들이 들어 있는 폴더 경로
IMAGE_DIR = r"D:\20251126_Pallet_Pose_images\Color"

# 2) 출력 폴더 (바탕화면 아래 sam3result)
OUTPUT_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "sam3result")

# 3) 텍스트 프롬프트들
TEXT_PROMPTS: List[str] = [
    "blue pallet",   # 필요하면 여러 개 추가 가능
]

# 4) SAM3 마스크 threshold
MASK_PROB_THRESHOLD = 0.3  # 0~1, 값이 클수록 마스크가 더 깐깐해짐

# 5) YOLO 클래스 ID (한 종류만 라벨링한다면 0 고정)
CLASS_ID = 0

# 6) polygon 최소 포인트 수 (너무 작은 잡음 제거용)
MIN_POLY_POINTS = 32


# =========================================
# SAM3 유틸
# =========================================

def load_sam3() -> Sam3Processor:
    print("[INFO] Loading SAM3 model ...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("[INFO] SAM3 model + processor loaded.")
    return processor


def run_sam3_text_prompts(
    processor: Sam3Processor,
    image_pil: Image.Image,
    text_prompts: List[str],
):
    """
    한 장의 PIL 이미지에 대해 여러 텍스트 프롬프트 실행.
    반환: [{ "prompt", "masks", "scores" }, ...]
    """
    inference_state = processor.set_image(image_pil)
    results = []

    for prompt in text_prompts:
        print(f"  [INFO] Running text prompt: '{prompt}'")
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        masks = output["masks"].detach().cpu()
        scores = output["scores"].detach().cpu()
        print(f"         -> Detected {masks.shape[0]} mask(s)")
        results.append({"prompt": prompt, "masks": masks, "scores": scores})

    return results


def masks_tensor_to_list_of_binary(
    masks_tensor,
    prob_threshold: float,
) -> List[np.ndarray]:
    """
    SAM3 masks 텐서를 개별 바이너리 마스크 리스트로 변환.
    각 마스크 shape: [H, W], 값 {0,1}
    """
    if masks_tensor.ndim == 4:
        # [N, 1, H, W] -> [N, H, W]
        masks_np = masks_tensor[:, 0, :, :].numpy()
    elif masks_tensor.ndim == 3:
        # [N, H, W]
        masks_np = masks_tensor.numpy()
    else:
        raise ValueError(f"Unexpected masks shape: {masks_tensor.shape}")

    bin_list: List[np.ndarray] = []
    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        binary = (m > prob_threshold).astype(np.uint8)
        if binary.sum() == 0:
            continue
        bin_list.append(binary)

    return bin_list


# =========================================
# YOLO polygon 변환 유틸
# =========================================

def binary_mask_to_normalized_polygon(
    mask_hw: np.ndarray,
    img_w: int,
    img_h: int,
    min_points: int = MIN_POLY_POINTS,
) -> Optional[List[float]]:
    """
    0/1 바이너리 마스크 → 가장 큰 컨투어 하나를 골라
    [x1, y1, x2, y2, ...] (0~1 정규화) 리스트로 반환.
    """
    if mask_hw is None or mask_hw.sum() == 0:
        return None

    # OpenCV는 0/255 마스크를 선호
    mask_uint8 = (mask_hw > 0).astype(np.uint8) * 255

    # 외곽선 추출
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 가장 큰 컨투어 선택
    contour = max(contours, key=cv2.contourArea)

    if contour.shape[0] < min_points:
        return None

    # (N,1,2) -> (N,2)
    contour = contour.reshape(-1, 2)

    poly_norm: List[float] = []
    for (x, y) in contour:
        xn = float(x) / float(img_w)
        yn = float(y) / float(img_h)
        poly_norm.append(xn)
        poly_norm.append(yn)

    return poly_norm


def save_yolo_seg_txt(
    txt_path: str,
    polygons: List[List[float]],
    class_id: int = CLASS_ID,
):
    """
    여러 polygon을 YOLO 세그멘테이션 형식으로 txt에 저장.
    한 polygon당 한 줄: class_id x1 y1 x2 y2 ...
    """
    lines: List[str] = []
    for poly in polygons:
        if not poly:
            continue
        # 소수점은 6자리 정도로 제한
        coord_str = " ".join(f"{v:.6f}" for v in poly)
        line = f"{class_id} {coord_str}"
        lines.append(line)

    if not lines:
        print(f"  [INFO] No polygons to write for {os.path.basename(txt_path)} (skip file).")
        return

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Wrote label: {txt_path}")


# =========================================
# 메인 로직
# =========================================

def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def main():
    if not os.path.isdir(IMAGE_DIR):
        raise NotADirectoryError(f"IMAGE_DIR not found: {IMAGE_DIR}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"[INFO] IMAGE_DIR   : {IMAGE_DIR}")
    print(f"[INFO] OUTPUT_ROOT : {OUTPUT_ROOT}")

    # 이미지 목록 수집
    filenames = sorted(f for f in os.listdir(IMAGE_DIR) if is_image_file(f))
    if not filenames:
        print("[WARN] No image files found in IMAGE_DIR.")
        return

    print(f"[INFO] Found {len(filenames)} image(s).")

    processor = load_sam3()

    for idx, fname in enumerate(filenames, 1):
        img_path = os.path.join(IMAGE_DIR, fname)
        print(f"\n[IMAGE {idx}/{len(filenames)}] {img_path}")

        # 이미지 로드
        image_pil = Image.open(img_path).convert("RGB")
        w, h = image_pil.size

        # SAM3 텍스트 프롬프트 실행
        results = run_sam3_text_prompts(
            processor=processor,
            image_pil=image_pil,
            text_prompts=TEXT_PROMPTS,
        )

        all_polygons: List[List[float]] = []

        # 프롬프트별 결과에서 마스크 → polygon 변환
        for res in results:
            masks = res["masks"]
            scores = res["scores"]
            prompt = res["prompt"]

            print(f"    prompt={prompt!r}, masks_shape={masks.shape}, scores={scores.tolist()}")

            bin_masks = masks_tensor_to_list_of_binary(
                masks_tensor=masks,
                prob_threshold=MASK_PROB_THRESHOLD,
            )

            for i, bm in enumerate(bin_masks):
                poly = binary_mask_to_normalized_polygon(
                    mask_hw=bm,
                    img_w=w,
                    img_h=h,
                    min_points=MIN_POLY_POINTS,
                )
                if poly is None:
                    continue
                all_polygons.append(poly)

        # txt 저장
        base, _ = os.path.splitext(fname)
        txt_name = base + ".txt"
        txt_path = os.path.join(OUTPUT_ROOT, txt_name)

        save_yolo_seg_txt(
            txt_path=txt_path,
            polygons=all_polygons,
            class_id=CLASS_ID,
        )

    print("\n[DONE] All images processed.")


if __name__ == "__main__":
    main()
