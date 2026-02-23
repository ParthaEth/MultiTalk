
export WAN_DISABLE_FLASH_ATTN="1"

python generate_multitalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --input_json /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/old-base.json \
    --sample_steps 8 \
    --mode streaming \
    --num_persistent_param_in_dit 30 \
    --audio_mode tts \
    --audio_save_dir local_data/sales_test
