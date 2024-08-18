exp_name="multi"
inference_file='../data/test/webshop_test.json'
num_processes='4'
weight_decay="0"

model_name="my_llama3_instruct"
### Default variables
task_name="webshop"
output_dir="../eval_outputs/${exp_name}/${model_name}/${task_name}_run2"
config_file="../ds_config/default_config_deepspeed_ga2.yaml"

# agent model
model_path="/workspace/multi_agent/AgentGym/agentenv/examples/bc_outputs/behavioral_clone_webshop_646/train_epoch_1"
model_type="llama3"

eval_batch_size="1"
num_workers="8"
seed="42"
do_sample="False"
temperature="1.0"


max_round="30"
env_server_base="http://127.0.0.1:36001"
data_len="200"
timeout="600"


#########
mkdir -p "${output_dir}"

CUDA_VISIBLE_DEVICES=3,4,5,6 \
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