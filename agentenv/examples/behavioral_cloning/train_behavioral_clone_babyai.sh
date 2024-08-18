exp_name="behavioral_clone_babyai_810"

n_epochs='3'

# accelerator config
num_processes='8'
main_process_port='8456'
config_file="../ds_config/default_config_deepspeed_ga2.yaml"

# training arguments
train_file='../data/single_env/babyai_810.json'
model_type="llama3"
model_train_path="/workspace/Llama-2-7b-chat-hf"
model_save_path="../bc_outputs/${exp_name}/"

batch_size="2"
eval_batch_size="1"
gradient_accumulation_steps="2"
max_input_length="4096"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
evaluating_epoch_freq="100"
saving_epoch_freq="3"
logging_step_freq="5"

# wandb config
wandb_log="True"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="200"
timeout="2400"

task_list=("webshop" "alfworld" "textcraft" "sciworld" "sqlgym" "lmrlgym_wordle" "lmrlgym_maze" "babyai" "weather" "movie" "todo" "academia" "sheet" "webarena")

# eval parameters
test_file_list=(
    "../data/test/webshop_test.json"
    "../data/test/alfworld_test.json"
    "../data/test/textcraft_test.json"
    "../data/test/sciworld_test_small.json"
    "../data/test/sqlgym_test_small.json"
    "../data/test/wordle_test.json"
    "../data/test/maze_test.json"
    "../data/test/babyai_test.json"
    "../data/test/tool_weather_test.json"
    "../data/test/tool_movie_test.json"
    "../data/test/tool_todo_test.json"
    "../data/test/tool_academia_test.json"
    "../data/test/tool_sheet_test.json"
    "../data/test/webarena_test.json"
)

do_sample="False"
temperature="1.0"
max_round_list=("10" "30" "20" "30" "1" "8" "15" "20" "10" "12" "15" "12" "15" "25")
env_server_base_list=(
    "http://127.0.0.1:36001"
    "http://127.0.0.1:36002"
    "http://127.0.0.1:59221"
    "http://127.0.0.1:59313"
    "http://127.0.0.1:59218"
    "http://127.0.0.1:59323/wordle"
    "http://127.0.0.1:59322/maze"
    "http://127.0.0.1:36003"
    "http://127.0.0.1:59213"
    "http://127.0.0.1:59325"
    "http://127.0.0.1:59327"
    "http://127.0.0.1:59326"
    "http://127.0.0.1:59217"
    "http://127.0.0.1:59340"
)

mkdir -p "${model_save_path}"
# # step1: train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    train_behavioral_clone.py \
        --train_file "${train_file}" \
        --inference_file "${test_file_list[0]}" \
        --test_file "${test_file_list[0]}" \
        --model_train_path "${model_train_path}" \
        --template_name "${model_type}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[0]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[0]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --env_server_base "${env_server_base_list[0]}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"\
        > "${model_save_path}/train.log" 2>&1

# step2: eval on test dataset
cur_task=${task_list[7]}
test_file=${test_file_list[7]}
max_round=${max_round_list[7]}
env_server_base=${env_server_base_list[7]}
eval_output_file="${model_save_path}/eval_${cur_task}.jsonl"

accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    ../../utils/distributed_eval_task.py \
        --model_path "${model_save_path}/train_epoch_${n_epochs}" \
        --output_file "${eval_output_file}" \
        --inference_file "${test_file}" \
        --task_name "${cur_task}" \
        --eval_batch_size "${eval_batch_size}" \
        --num_workers "${num_workers}" \
        --seed "${seed}" \
        --do_sample "${do_sample}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"  \
        > ${model_save_path}/eval.log 2>&1
