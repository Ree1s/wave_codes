CUDA_VISIBLE_DEVICES=0 python preprocess_temp.py \
--data_path data/woman-running.mp4 \
--H 512 \
--W 512 \
--save_dir latents_group_random \
--sd_version '1.5' \
--steps 50 \
--batch_size 8 \
--save_steps 50 \
--n_frames 40 \
--inversion_prompt 'A woman is running'