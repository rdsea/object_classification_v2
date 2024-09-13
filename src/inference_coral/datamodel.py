from enum import Enum

from pydantic import BaseModel


class ImageClassificationModelEnum(Enum):
    EFFICIENTNET_EDGETPU_L_QUANT = "efficientnet-edgetpu-L_quant"
    EFFICIENTNET_EDGETPU_M_QUANT = "efficientnet-edgetpu-M_quant"
    EFFICIENTNET_EDGETPU_S_QUANT = "efficientnet-edgetpu-S_quant"
    INCEPTION_V1_224_QUANT = "inception_v1_224_quant"
    INCEPTION_V2_224_QUANT = "inception_v2_224_quant"
    INCEPTION_V3_299_QUANT_TFLITE = "inception_v3_299_quant.tflite"
    INCEPTION_V4_299_QUANT = "inception_v4_299_quant"
    TF2_MOBILENET_V1_1_0_224_PTQ = "tf2_mobilenet_v1_1.0_224_ptq"
    TF2_MOBILENET_V2_1_0_224_PTQ = "tf2_mobilenet_v2_1.0_224_ptq"
    TFHUB_TF2_RESNET_50_IMAGENET_PTQ = "tfhub_tf2_resnet_50_imagenet_ptq"


class ModelConfig(BaseModel):
    input_shape: tuple[int, int, int, int]
    input_mode: str


class InferenceServiceConfig(BaseModel):
    pipeline_id: str
    ensemble: bool
    model_config_dict: dict[ImageClassificationModelEnum, ModelConfig]
    external_services: dict
