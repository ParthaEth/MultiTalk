import os

AVATAR_DIR = '/home/web/partha/vidLink/video_generators/multitalk/Input_outputs/input_files/sales_executive'
CKPT_DIR = './weights/Wan2.1-I2V-14B-480P'
WAV2VEC_DIR = './weights/chinese-wav2vec2-base'
KOKORO_DIR = './weights/Kokoro-82M'
TTS_VOICE = './weights/Kokoro-82M/voices/af_heart.pt'
# Short prompt: detailed scene text hurts lip sync with FusionX.
VIDEO_PROMPT = os.getenv(
    "MULTI_TALK_VIDEO_PROMPT",
    "A person speaks to the camera.",
)
# FusionX LoRA (bf16 path). Set MULTI_TALK_LORA_DIR="" to disable.
LORA_DIR = os.getenv(
    "MULTI_TALK_LORA_DIR",
    "./weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
)
LORA_SCALE = float(os.getenv("MULTI_TALK_LORA_SCALE", "1.0"))
SAMPLE_STEPS = int(os.getenv("MULTI_TALK_SAMPLE_STEPS", "8"))
SAMPLE_SHIFT = float(os.getenv("MULTI_TALK_SAMPLE_SHIFT", "2"))
# FusionX: low text CFG; higher audio CFG for stronger lip sync.
SAMPLE_TEXT_GUIDE_SCALE = float(os.getenv("MULTI_TALK_SAMPLE_TEXT_GUIDE_SCALE", "1.0"))
SAMPLE_AUDIO_GUIDE_SCALE = float(os.getenv("MULTI_TALK_SAMPLE_AUDIO_GUIDE_SCALE", "5.0"))
# TeaCache can hurt lip sync with FusionX 2-NFE CFG; off by default.
USE_TEACACHE = os.getenv("MULTI_TALK_USE_TEACACHE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Time required to generate 1 second of video (in seconds).
# For multitalk: 4 minutes per video second = 240 seconds
TIME_PER_VIDEO_SECOND_SECONDS = 300.0
