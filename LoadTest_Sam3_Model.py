from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("Loading SAM3 model...")

# SAM3 이미지 모델 생성
model = build_sam3_image_model()

# 프로세서 생성 (기본값으로)
processor = Sam3Processor(model)

print("✅ SAM3 model + processor loaded OK")