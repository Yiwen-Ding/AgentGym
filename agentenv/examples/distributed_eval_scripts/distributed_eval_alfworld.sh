exp_name="unseen"
inference_file='../data/unseen/alfworld_unseen_20.json'

num_processes='4'
main_process_port='8877'
weight_decay="0"

### Default variables
task_name="alfworld"
output_dir="../eval_outputs/${exp_name}/${task_name}"
config_file="../ds_config/default_config_deepspeed_ga2.yaml"

# agent model
# model_path="/workspace/AgentEvol-7B"
model_path="/workspace/agentlm-7b"
model_type="llama2"

eval_batch_size="1"
num_workers="8"
seed="43"
do_sample="False"
temperature="1.0"

max_round="30"
env_server_base="http://127.0.0.1:36001"
data_len="200"
timeout="2400"


#########
mkdir -p "${output_dir}"

accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
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