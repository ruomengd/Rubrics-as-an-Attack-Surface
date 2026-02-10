#! /bin/bash

cd ./data


# Download RMB Harmlessness dataset
mkdir -p ./raw_data/RMB-Harmlessness/
git clone https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.git
mv RMB-Reward-Model-Benchmark/RMB_dataset/Pairwise_set/Harmlessness/* ./raw_data/RMB-Harmlessness/
rm -rf RMB-Reward-Model-Benchmark


# Helpfulness
## LM Arena
python scripts/helpfulness/data_lmarena.py --max-total-tokens 4096 --n-train 1000 --n-val 1000 --n-test 1000 --seed 42 --out-prefix raw_data
## Ultra
python scripts/helpfulness/data_ultra.py --max-total-tokens 4096 --n-train 1000 --n-val 1000 --n-test 1000 --seed 42 --out-prefix raw_data
## Split Helpfulness into Splits (Bench and Target)
python scripts/helpfulness/data_split_domain.py --raw_data_dir ./raw_data --out-dir ./helpfulness


# Harmlessness
## SafeRLHF-RMB
python scripts/harmlessness/data_SafeRLHF-RMB.py --max-total-tokens 1024 --n-train 1000 --n-val 1000 --n-test 1000 \
        --seed 42 --out-prefix harmlessness --bench_domain SafeRLHF --target_domain RMB

## RMB-SafeRLHF
python scripts/harmlessness/data_SafeRLHF-RMB.py --max-total-tokens 1024 --n-train 1000 --n-val 1000 --n-test 1000 \
        --seed 42 --out-prefix harmlessness --bench_domain RMB --target_domain SafeRLHF

## Anthropic-SafeRLHF (+dpo: used for downstream policy misalignment experiments)
python scripts/harmlessness/data_Anthropic-SafeRLHF.py --max-total-tokens 1024 --n-train 1000 --n-val 1000 --n-test 1000 --is_dpo \
        --seed 42 --out-prefix harmlessness --bench_domain Anthropic --target_domain SafeRLHF

## SafeRLHF-Anthropic
python scripts/harmlessness/data_Anthropic-SafeRLHF.py --max-total-tokens 1024 --n-train 1000 --n-val 1000 --n-test 1000 \
        --seed 42 --out-prefix harmlessness --bench_domain SafeRLHF --target_domain Anthropic