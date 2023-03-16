
export PYTHONPATH=".:guided-diffusion:improved-diffusion"
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
# python scripts/image_sample.py --image_size 256 --num_channels 128 --num_res_blocks 3 \
# --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine \
# --batch_size 1000 --num_samples 50001 --timestep_respacing 100 --use_ddim False \
# --model_path  "/home/yzh/docs/pytorch/PTQDiffusionModel/guided-diffusion/models/256x256_diffusion_uncond.pt"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
QUANT_FLAGS="--n_bits_w 8 --channel_wise --n_bits_a 8  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5 --iters_w 100 --calib_num_samples 128 \
--data_dir /datasets/imagenet --calib_im_mode random"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --timestep_respacing 250 --use_ddim True --noise_schedule cosine"

BATCH_SIZE=600
NUM_SAMPLES=10000

 export CUDA_VISIBLE_DEVICES="1"
 CALIB_FLAGS="--calib_t_mode random --out_path outputs_mixup_quant_on_cosine/random8-5bit.npz"
 python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path /home/shangyuzhang/diffusion_models/ptqdiffusionmodel/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &

# export CUDA_VISIBLE_DEVICES="1"
# CALIB_FLAGS="--calib_t_mode -1"
# python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &

# export CUDA_VISIBLE_DEVICES="2"
# CALIB_FLAGS="--calib_t_mode 1"
# python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &

#export CUDA_VISIBLE_DEVICES="1"
#CALIB_FLAGS="--calib_t_mode uniform"
#python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &


# export CUDA_VISIBLE_DEVICES="4"
# CALIB_FLAGS="--calib_t_mode mean" 
# python improved-diffusion/scripts/quant_image_sample.py $CALIB_FLAGS $QUANT_FLAGS $MODEL_FLAGS --model_path guided-diffusion/models/imagenet64_uncond_100M_1500K.pt $DIFFUSION_FLAGS --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE &

