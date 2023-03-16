export PYTHONPATH=".:guided-diffusion"
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
# python scripts/image_sample.py --image_size 256 --num_channels 128 --num_res_blocks 3 \
# --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine \
# --batch_size 1000 --num_samples 50001 --timestep_respacing 100 --use_ddim False \
# --model_path  "/home/yzh/docs/pytorch/PTQDiffusionModel/guided-diffusion/models/256x256_diffusion_uncond.pt"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
QUANT_FLAGS="--n_bits_w 8 --channel_wise --n_bits_a 8  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5 --iters_w 100 --calib_num_samples 128"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
python guided-diffusion/scripts/quant_image_sample.py $QUANT_FLAGS $MODEL_FLAGS --model_path guided-diffusion/models/64x64_diffusion.pt $SAMPLE_FLAGS --num_samples 10 --batch_size 16
# python -m cProfile -o quant.pstats scripts/quant_image_sample.py $QUANT_FLAGS $MODEL_FLAGS --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS --num_samples 100 --batch_size 16
