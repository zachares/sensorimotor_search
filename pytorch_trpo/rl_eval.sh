#!/bin/bash

# echo First:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_32_.ckpt.20190322181301364216.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/32_sim/run_0/journal_32_simitr_2905.ckpt"
# Z_DIM="32"
# RUN_NOTES="zdim32_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/
# wait 

# echo Second:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_64_.ckpt.20190322181249491245.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/64_sim/journal_zdim_64_trial_6itr_3555.ckpt"
# Z_DIM="64"
# RUN_NOTES="zdim64_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/
# wait 

# echo Third:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_no_vision_.ckpt.20190328161240523280.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/no_vision/run_3/journal_no_visionitr_1255.ckpt"
# Z_DIM="128"
# RUN_NOTES="no_vision128_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/ --vision 0.0
# wait 

# echo Third:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_no_pairing_.ckpt.20190402103434460767.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/no_pairing/run_4/journal_nopairing_zdim_128_trial0itr_55.ckpt"
# Z_DIM="128"
# RUN_NOTES="no_pairing128_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/
# wait 

# echo Fourth:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_no_depth_.ckpt.20190328161407087656.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/no_depth/run_2/journal_no_depth_contitr_1305.ckpt"
# Z_DIM="128"
# RUN_NOTES="no_depth128_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/ --depth 0.0
# wait 

echo Fifth:

REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_.ckpt.20190401161143969508.20"
RL_MODEL="/juno/group/multimodal_project/rl_model/deterministic/run_3/journal_deterministic_zdim_128_trial0itr_1005.ckpt"
Z_DIM="128"
RUN_NOTES="deterministic128_eval"

python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/
wait 


# echo First:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_no_vision_no_depth_no_force_.ckpt.20190331135826594739.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/no_vision_depth_force/run_2/journal__no_force_vision_depth1itr_3505.ckpt"
# Z_DIM="128"
# RUN_NOTES="no_vision_force_depth128_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/ --vision 0.0 --force 0.0 --depth 0.0
# wait 

# echo Second:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_128_no_force_.ckpt.20190331135752509132.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/no_force/run_1/journal__noforce_sim1itr_1705.ckpt"
# Z_DIM="128"
# RUN_NOTES="no_force128_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/ --force 0.0
# wait 

# echo Eighth:

# REP_MODEL="/juno/group/multimodal_project/rep_model/journal_selfsupervised_zdim_256_variational_.ckpt.20190409142134559092.20"
# RL_MODEL="/juno/group/multimodal_project/rl_model/256_sim/journal_zdim_256_itr_4255.ckpt"
# Z_DIM="256"
# RUN_NOTES="zdim256_eval"

# python main.py --env-name multimodal --action-dim 4 --box-reset-config /scr-ssd/pytorch-trpo/goal_conditioned.yaml --controller POS_YAW --z-dim $Z_DIM --rep-model $REP_MODEL --run-notes $RUN_NOTES --load $RL_MODEL  --eval-eps 50 --eval-interval 1 --log-dir /juno/group/multimodal_project/eval_results/
