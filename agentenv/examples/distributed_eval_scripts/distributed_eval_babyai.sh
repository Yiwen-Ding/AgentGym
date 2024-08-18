exp_name="babyai"
inference_file='../data/test/babyai_test.json'

num_processes='2'
weight_decay="0"

model_name="agentlm-7b"
### Default variables
task_name="babyai"
output_dir="../eval_outputs/${exp_name}/${model_name}/${task_name}"
config_file="../ds_config/default_config_deepspeed_ga2.yaml"
# agent model
model_path="/workspace/dyw/AgentGym/agentenv/examples/bc_outputs/behavioral_clone_webshop_3930/train_epoch_3"
model_type="llama3"

eval_batch_size="1"
num_workers="8"
seed="42"
do_sample="False"
temperature="1.0"


max_round="20"
env_server_base="http://127.0.0.1:36001"
data_len="200"
timeout="600"


#########
mkdir -p "${output_dir}"

CUDA_VISIBLE_DEVICES=2,3 \
accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
    distributed_eval_task.py \
            --model_path "${model_path}" \
            --output_file "${output_dir}/eval_${task_name}.jsonl" \
            --inference_file "${inference_file}" \
            --task_name "${task_name}" \
            --eval_batch_size "${eval_batch_size}" \
            --num_workers "${num_workers}" \
            --seed "${seed}" \
            --do_sample "${do_sample}" \
            --temperature "${temperature}" \
            --max_round "${max_round}" \
            --env_server_base "${env_server_base}" \
            --data_len "${data_len}" \
            --timeout "${timeout}" \
            --template_name "${model_type}" \
            > "${output_dir}/eval_${task_name}.log" 2>&1