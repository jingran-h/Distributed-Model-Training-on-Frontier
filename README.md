# Distributed-Model-Training-on-Frontier

This repository contains a supervised fine-tuning (SFT) walkthrough for running LoRA-based training of Meta-Llama-3 on ORNL's Frontier. The instructions below assume you have been granted access to Frontier, an approved Hugging Face token for the model, and sufficient scratch space.

Throughout the guide we will rely on a scratch root. Scratch provides the largest quotas on Frontier, so keep bulky artifacts (models, datasets, checkpoints) there rather than in home or project spaces:

```bash
export SCRATCH_ROOT=/lustre/orion/<allocation>/scratch/$USER
```

Replace `<allocation>` with your actual project identifier (for example, `gen150`, `cfd204`, etc.). Every time you copy a sample command, make sure the allocation prefix matches your project.

The training scripts read model, tokenizer, and dataset snapshots from

```
$SCRATCH_ROOT/llama-models-cache
$SCRATCH_ROOT/llama-tokenizers-cache
$SCRATCH_ROOT/data_cache
```

If you prefer a different layout, update `SCRATCH_ROOT` at the top of the Python scripts (`sft_llama_ds.py`, `sft_llama_deepspeed.py`, `sft_llama.py`, etc.).

---

## 1. Prepare The Environment

Most setup (Conda environment creation, package installs, dataset/model downloads) should be performed on a **login node** to keep compute nodes free for training. Use the compute partition only when a tool requires ROCm runtime access.

### 1.1 Login node steps

On a login node:

```bash
module reset
module load gcc/12.2.0
module load miniforge3/23.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
```

Create and populate a ROCm-enabled Conda environment:

```bash
conda create -y -p $SCRATCH_ROOT/envs/frontier-ft python=3.10
conda activate $SCRATCH_ROOT/envs/frontier-ft
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install \
  accelerate==1.10.1 \
  datasets==3.1.0 \
  deepspeed==0.17.5 \
  evaluate==0.4.3 \
  peft==0.17.1 \
  safetensors==0.6.2 \
  sentencepiece==0.2.1 \
  tensorboard==2.20.0 \
  transformers==4.57.0 \
  trl==0.23.1
```

You can also download the model and dataset (Sections 2 and 3) entirely from the login node.

### 1.2 Compute node steps

ROCm 6.1.3 is only available on compute nodes. For testing or running the training script interactively, request a compute node:

```bash
salloc -A <allocation> -t 00:30:00 -N 1
```

On the compute node:

```bash
module reset
module load rocm/6.1.3
module load gcc/12.2.0
module load miniforge3/23.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH_ROOT/envs/frontier-ft
module unload miniforge3/23.11.0
```

Set up paths so the environment can build DeepSpeed extensions and find ROCm:

```bash
export ROCM_HOME=/opt/rocm-6.1.3
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH_ROOT/deepspeed_extensions}
export CC=$(which gcc)
export CXX=$(which g++)
mkdir -p "$TORCH_EXTENSIONS_DIR"
```

DeepSpeed uses `~/.deepspeed_env` to seed its launch environment. Regenerate it after loading the modules above (either login or compute node—just ensure the variables reflect the ROCm-enabled environment):

```bash
cat <<EOF > "$HOME/.deepspeed_env"
PATH=$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ROCM_HOME=$ROCM_HOME
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
CC=$CC
CXX=$CXX
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
EOF
```

> **Tip:** When you move to a compute node, unload Miniforge after the Conda environment has been activated: `module unload miniforge3/23.11.0`.

---

## 2. Download Model & Tokenizer Snapshots (login node)

Training runs on compute nodes do not have external network access. Download the Meta-Llama-3 model and tokenizer on a login node while the network is available.

```bash
export HF_TOKEN=<your_huggingface_token>
mkdir -p $SCRATCH_ROOT/llama-models-cache
mkdir -p $SCRATCH_ROOT/llama-tokenizers-cache

huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type model \
  --local-dir $SCRATCH_ROOT/llama-models-cache \
  --local-dir-use-symlinks False

huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type tokenizer \
  --local-dir $SCRATCH_ROOT/llama-tokenizers-cache \
  --local-dir-use-symlinks False
```

If you do not have `huggingface-cli` available, install it with `pip install huggingface_hub`.

---

## 3. Cache The Training Dataset (login node)

The example SFT uses `mlabonne/guanaco-llama2-1k`. Download it ahead of time on a login node:

```bash
mkdir -p $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k
huggingface-cli download mlabonne/guanaco-llama2-1k \
  --repo-type dataset \
  --local-dir $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k \
  --local-dir-use-symlinks False
```

The training script looks for the Arrow file `guanaco-llama2-1k-train.arrow` under that directory. Confirm the file exists before moving on.

To keep the run fully offline on compute nodes, export:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 4. Configure The Training Job

- Adjust the job options in `launch_deepspeed_ft.slurm` to match your allocation (`-A`, walltime, partition, node count, etc.). Replace `GEN150` with your project ID everywhere it appears and update the output/error log paths if you store them elsewhere.
- Update storage paths in the SLURM script so they resolve for your account (e.g., the scratch prefix, TORCH extensions directory, and any temporary file locations).
- Confirm `SCRATCH_ROOT` inside the Python scripts points to your scratch directory. The default is `/lustre/orion/<allocation>/scratch/$USER`; edit `sft_llama_ds.py` (and related scripts) if your layout differs.
- Ensure `HF_TOKEN` (and any proxy variables) are set inside the SLURM script if the model requires authentication, even when cached.

---

## 5. Submit The Demo SFT Run

1. Activate the environment on the login node (or an interactive batch session):

   ```bash
   module reset
   module load rocm/6.1.3
   module load gcc/12.2.0
   module load miniforge3/23.11.0
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate $SCRATCH_ROOT/envs/frontier-ft
   module unload miniforge3/23.11.0

   export HF_HUB_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

2. Submit the batch job:

   ```bash
   sbatch launch_deepspeed_ft.slurm
   ```

3. Monitor progress:

   ```bash
   tail -f logs/finetune_llama3_ds-<jobid>.o
   ```

   You should see output confirming that the training data is loaded from the local cache and that the DeepSpeed fused-adam extension builds successfully.

---

## 6. Inspect Results

When the run completes, checkpoints land under `results_modified/`. The final LoRA adapter is in the newest `checkpoint-*` directory—for example:

```
results_modified/checkpoint-63/adapter_model.safetensors
```

Use `zero_to_fp32.py` (inside the checkpoint directory) to convert the sharded weights to a full-precision model if required:

```bash
python results_modified/checkpoint-63/zero_to_fp32.py \
  --model_dir results_modified/checkpoint-63 \
  --merge_lora \
  --output_path finetuned_llama3_8B.safetensors
```

Point downstream inference scripts at `results_modified/checkpoint-63` (or whichever checkpoint you want to evaluate).

---

## 7. Troubleshooting

- **DeepSpeed complains about missing ROCm/CUDA:** Re-run the module loads and regenerate `~/.deepspeed_env` so worker nodes inherit `ROCM_HOME`, `PATH`, and `LD_LIBRARY_PATH`.
- **Space constraints:** The Meta-Llama-3-8B snapshot requires ~15 GB. Verify quota before downloading.

Following the steps in this tutorial gives you a reproducible SFT workflow on Frontier without relying on user-specific paths or live network connectivity during the batch job. Adjust hyperparameters in `sft_llama_ds.py` to experiment with longer runs, different datasets, or alternative LoRA settings.
