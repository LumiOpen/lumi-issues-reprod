# GRPO with TRL

GRPO is a reinforcement learning algorithm popularized by DeepSeek math and reasoning models. GRPO has been implemented in various ML frameworks, the most widely used being [HuggingFace TRL](https://huggingface.co/docs/trl/en/index). TRL has dependencies on libraries such as [accelerate](https://huggingface.co/docs/accelerate/index), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) and [vLLM](https://github.com/vllm-project/vllm).

We wish to use TRL for GRPO (and other) post-training experiments, but have faced various issues while using this library on Lumi.

For us to deem TRL functional on Lumi, the following requirements must be met:
- We can replicate a simple GRPO example from the TRL documentation without numerical issues and acceptable throughput
- We can replicate more complex reasoning recipes from [Open R1](https://github.com/huggingface/open-r1)
- We can utilize vLLM with considerable throughput gains
- We can scale to multiple nodes with improved througput
- We can scale to 70B models

TODO: Add replication instructions for all requirements

## A simple GRPO example

TRL documentation provides a simple GRPO example, which we have in `train_grpo.py`

The documentation shows stable training with rewards and loss increasing (this is to be expected with GRPO):
![GRPO curves](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_curves.png)

The TRL documentation states "Distributed across 8 GPUs, the training takes approximately 1 day."
We assume this refers to H100s and we do not expect to reach similar performance, but this establishes the only throughput baseline we can compare against.

Our replication of this simple example can be done with `train_grpo.slurm`

Simply run

    sbatch train_grpo.slurm

### Observations and issues:
- We do not know whether Qwen model family works correctly on Lumi, due to this warning: Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
- With the default hyperparameters the training results in **NaN** grad norm already on the first or second training step:
    {'loss': 0.0438, 'grad_norm': `nan`, 'learning_rate': 9.997715486715555e-07, 'num_tokens': 372082.0, 'completion_length': 254.678125, 'rewards/reward_len': -389.834375, 'reward': -389.834375, 'reward_std': 28.132754516601562, 'kl': `nan`, 'clip_ratio': 0.0, 'epoch': 0.0}
- If we reduce `per_device_train_batch_size` to 1, training seems numerically stable, but training time estimation varies between 4 and 30 days. Training time is hard to estimate, since most of the compute is used by inference and generation length affects this drastically.
- With DeepSpeed Zero stage 3, estimated training time is over 10 days. Is this to be expected?