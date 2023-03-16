export PYTHONPATH=".:guided-diffusion:improved-diffusion"

QUANT_FLAGS="--n_bits_w 8 --channel_wise --n_bits_a 8  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5 --iters_w 100 --calib_num_samples 1024 \
--data_dir /datasets/imagenet --calib_im_mode noise_backward_t"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 250 --use_ddim True --noise_schedule cosine"

BATCH_SIZE=500
NUM_SAMPLES=10000

export CUDA_VISIBLE_DEVICES="7"
CALIB_FLAGS="--calib_t_mode normal --calib_t_mode_normal_mean 0.4 --calib_t_mode_normal_std 0.4 --out_path /home/shangyuzhang/diffusion_models/PTQ4DM/results/random8-normalmean04std040_ddim250.npz"
#python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path /pretrained-model-path/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &
python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path /home/shangyuzhang/diffusion_models/ptqdiffusionmodel/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &
