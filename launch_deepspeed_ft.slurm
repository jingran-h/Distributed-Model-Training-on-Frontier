#!/bin/bash

#SBATCH -A GEN150
#SBATCH -J finetune_llama3_ds 
#SBATCH -o logs/finetune_llama3_ds-%j.o
#SBATCH -e logs/finetune_llama3_ds-%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --gres=gpu:8
#SBATCH -N 2                        # number of nodes for multi-node training 


module reset
module load rocm/6.1.3
module load gcc/12.2.0
module load miniforge3/23.11.0

export ROCM_HOME=/opt/rocm-6.1.3
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:${LD_LIBRARY_PATH:-}

# ensure recent toolchain is picked up when compiling DS ops
export CC=$(which gcc)
export CXX=$(which g++)

# place DS build artifacts in scratch to avoid reusing stale system-wide binaries
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/lustre/orion/gen150/scratch/william_huang/deepspeed_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"

# refresh DeepSpeed launcher environment so worker nodes inherit the right toolchains
cat <<EOF > "$HOME/.deepspeed_env"
PATH=$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ROCM_HOME=$ROCM_HOME
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
CC=$CC
CXX=$CXX
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EOF

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /lustre/orion/gen150/scratch/william_huang/envs/frontier-ft
module unload miniforge3/23.11.0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

export HF_TOKEN="YOUR_HF_TOKEN"

# Disable AMD SMI to avoid compatibility issues
export DISABLE_AMDSMI=1

scontrol show hostnames $SLURM_NODELIST > job.node.list.$SLURM_JOB_ID
input="./job.node.list".$SLURM_JOB_ID
host_file=host_file.$SLURM_JOB_ID
readarray -t arr <"$input"

for item in "${arr[@]}"; do
  echo "$item" slots=8 >> $host_file 
done

first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

#torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=$MASTER_ADDR \
#--master_port=29500 sft_llama_deepspeed.py


# Multi-node DeepSpeed launch
deepspeed --num_gpus 8 --num_nodes 2 --hostfile $host_file --master_addr=$MASTER_ADDR --master_port=29500 \
./sft_llama_ds.py
