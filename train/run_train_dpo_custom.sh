set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34230
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#export NCCL_IB_SL=3
#export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO

OUTPUT_DIR=/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-custom_merge_sharegpt-alpaca-gpt
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
unset TRANSFORMERS_CACHE
unset HF_MODULES_CACHE
unset TORCH_EXTENSIONS_DIR
unset TORCH_HOME
unset TRITON_CACHE_DIR
unset HF_HOME

pip uninstall -y apex
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com torchvision==0.18.0
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com  torch==2.3.0
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com /nfs-151/disk8/townswu/codes/VITA-main/script/whl_list/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com transformers==4.51.0
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com trl

pip uninstall apex -y
#cd /var/s3fs/wutong02/codes/apex-master
#pip install -v --no-cache-dir --global-option=“–cpp_ext” --global-option=“–cuda_ext” ./
#cd /home/storages/dev95/disk8/townswu/codes/InternVL-main/internvl_chat/
#pip install apex
pip list

cd /nfs-151/disk8/townswu/codes/qwen3_emoji_dpo/
BATCH_SIZE=4
EPOCH=1
GPUS=2
torchrun \
  --nnodes=$WORLD_SIZE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=$GPUS \
  --master_port=${MASTER_PORT} \
  ./train/train_dpo_custom.py \
  --train_file /nfs-151/disk8/townswu/codes/qwen3_emoji_dpo/data/dpo_emoji_english_merge_sharegpt-alpaca-gpt.jsonl \
  --model_name /nfs-151/disk6/townswu/pretrained_models/Qwen/Qwen3-0.6B \
  --output_dir ${OUTPUT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCH} \
  --gradient_accumulation_steps 1 \
  --lr 5e-6 \
  --bf16 \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
