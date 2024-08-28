python3 benchmark_throughput.py --input_len 128 --output_len 128 --model /data/project/EXAONE_v3.0/model_7.8B_4k_beta_2024-07-30 --trust_remote_code --dataset "/data/project/vllm/benchmarks/magpie_pro_mt_300k.json"

python3 benchmark_throughput.py --model /data/project/EXAONE_v3.0/model_7.8B_4k_beta_2024-07-30 --trust_remote_code --dataset "ShareGPT_V3_unfiltered_cleaned_split.json"
python3 benchmark_throughput.py --model /data/project/EXAONE_v3.0/model_7.8B_4k_beta_2024-07-30 --trust_remote_code --dataset "magpie_pro_mt_300k.json"
python3 benchmark_throughput.py --model /data/project/EXAONE_v3.0/model_7.8B_4k_beta_2024-07-30 --trust_remote_code --dataset "sharegpt-tagengo-gpt4-ko.json"

### benchmark througput

CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model /data/project/EXAONE_v3.0/model_7.8B_4k_beta_2024-07-30 --trust_remote_code --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --trust_remote_code --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --gpu-memory-utilization 0.95 --max_model_len 4096 --download-dir /data/project/yohan/98_model --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --trust_remote_code --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o

CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model Qwen/Qwen1.5-1.8B --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model Qwen/Qwen2-0.5B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model Qwen/Qwen2-1.5B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model Qwen/Qwen2-7B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model google/gemma-2-2b-it --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model mistralai/Mistral-7B-Instruct-v0.3 --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
python3 benchmark_throughput.py --model google/gemma-2-9b-it --download-dir /data/project/yohan/98_model --max_model_len 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
python3 benchmark_throughput.py --model mistralai/Mistral-Nemo-Instruct-2407 --download-dir /data/project/yohan/98_model --max_model_len 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o

CUDA_VISIBLE_DEVICES=0 python3 benchmark_throughput.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --dataset "ShareGPT_V3_unfiltered_cleaned_split.json" --output_json o --quantization fp8

# python3 benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --download-dir /data/project/yohan/98_model --max_model_len 4096 --gpu-memory-utilization 0.95 --tensor-parallel-size 2 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
# python3 benchmark_throughput.py --model deepseek-ai/deepseek-moe-16b-chat --trust_remote_code --download-dir /data/project/yohan/98_model --max_model_len 4096 --tensor-parallel-size 2 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
python3 benchmark_throughput.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --download-dir /data/project/yohan/98_model --max_model_len 4096 --tensor-parallel-size 2 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model microsoft/Phi-3.5-mini-instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model meta-llama/Llama-2-13b-chat-hf --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throughput.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --max_model_len 4096 --dataset "sharegpt-tagengo-gpt4-ko.json" --output_json o


### benchmark serving
CUDA_VISIBLE_DEVICES=1 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 75
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 125
CUDA_VISIBLE_DEVICES=1 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 175
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 250
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 10
CUDA_VISIBLE_DEVICES=1 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 20
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 50
CUDA_VISIBLE_DEVICES=1 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 200
CUDA_VISIBLE_DEVICES=1 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 500
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 1000

python3 benchmark_serving.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen1.5-1.8B --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-0.5B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2-1.5B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-7B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-2-2b-it --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve google/gemma-2-9b-it --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.3 --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve mistralai/Mistral-Nemo-Instruct-2407 --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-2-7b-chat-hf --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-moe-16b-chat --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve microsoft/Phi-3.5-mini-instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --download-dir /data/project/yohan/98_model --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-2-13b-chat-hf --download-dir /data/project/yohan/98_model --port 8001 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100
CUDA_VISIBLE_DEVICES=0 vllm serve /data/project/EXAONE_v3.0/model_2.4B_4k_beta_2024-06-11 --port 8000 --trust-remote-code --max-model-len=4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-num-seqs 100

python3 benchmark_serving.py --model Qwen/Qwen1.5-1.8B --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model Qwen/Qwen2-0.5B-Instruct --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model Qwen/Qwen2-1.5B-Instruct --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model Qwen/Qwen2-7B-Instruct --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model google/gemma-2-2b-it --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model google/gemma-2-9b-it --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model mistralai/Mistral-7B-Instruct-v0.3 --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model mistralai/Mistral-Nemo-Instruct-2407 --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model meta-llama/Llama-2-7b-chat-hf --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model deepseek-ai/deepseek-moe-16b-chat --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model microsoft/Phi-3.5-mini-instruct --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model microsoft/Phi-3.5-MoE-instruct --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"
python3 benchmark_serving.py --model meta-llama/Llama-2-13b-chat-hf --port 8001 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"

python3 benchmark_serving.py --model /data/project/EXAONE_v3.0/model_2.4B_4k_beta_2024-06-11 --port 8000 --dataset-name sharegpt --dataset "sharegpt-tagengo-gpt4-ko.json" --save-result --result-dir "benchmark_output"

