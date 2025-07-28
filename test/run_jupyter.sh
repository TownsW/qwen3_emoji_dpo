set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34230
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO

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
pip install -i http://pip.duxiaoman-int.com/root/pypi/+simple/  --trusted-host pip.duxiaoman-int.com jupyter notebook

pip uninstall apex -y
pip list

jupyter notebook --port=8889