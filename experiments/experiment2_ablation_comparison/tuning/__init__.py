"""PPO调参模块（实验二最小方案）。"""

'''
conda run -n abm python experiments/experiment2_ablation_comparison/tuning/tune_ppo.py \
  --mode debug --coarse-trials 2 --refine-trials 1 \
  --coarse-episodes 100 --refine-episodes 150 --final-episodes 200 \
  --coarse-seeds 1 --refine-seeds 1 --final-seeds 2 --post-eval-episodes 10
  
conda run -n abm python experiments/experiment2_ablation_comparison/tuning/tune_ppo.py \
  --mode debug --coarse-trials 24 --refine-trials 12 \
  --coarse-episodes 300 --refine-episodes 600 --final-episodes 1000 \
  --coarse-seeds 1 --refine-seeds 3 --final-seeds 5 --post-eval-episodes 30 --seed-jobs 5

'''
