export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=".:guided-diffusion:improved-diffusion"
# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 2000 --use_ddim True --noise_schedule cosine"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# python guided-diffusion/scripts/image_sample.py $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples 100 --batch_size 100


# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# python improved-diffusion/scripts/image_sample.py $MODEL_FLAGS --model_path guided-diffusion/models/cifar10_uncond_50M_500K.pt $DIFFUSION_FLAGS --num_samples 10 --batch_size 10

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 250 --use_ddim True --noise_schedule cosine"
# python guided-diffusion/scripts/image_sample.py $MODEL_FLAGS --model_path guided-diffusion/models/cifar10_uncond_50M_500K.pt $DIFFUSION_FLAGS --num_samples 10 --batch_size 10
python improved-diffusion/scripts/image_sample.py $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples 10000 --batch_size 32