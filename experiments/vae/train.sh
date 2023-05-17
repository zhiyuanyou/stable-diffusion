export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1
python -u ../../main.py -b autoencoder_kl_8x8x64.yaml -t
