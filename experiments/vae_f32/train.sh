export PYTHONPATH=/opt/data/private/142/stable-diffusion/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
/root/anaconda3/envs/ldm/bin/python -u /opt/data/private/142/stable-diffusion/main.py -b autoencoder_kl_8x8x64.yaml -t > train.log
