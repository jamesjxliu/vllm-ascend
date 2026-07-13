#!/bin/bash

ROLE="prefill"              # prefill / decode
HARDWARE_SERIES="A3"        # A2 (800I/800T A2) or A3 (800I/800T A3)
LOCAL_IP="80.5.17.34"
NIC_NAME="enp194s0f0"

export VLLM_ASCEND_ZBAL_LOCAL_MEM_SIZE=58368
export VLLM_ASCEND_ZBAL_BOOTSTRAP_URL="tcp://80.5.17.34:16989"
export VLLM_ASCEND_ZBAL_MOE_ENABLE=1
export VLLM_ASCEND_ZBAL_MOE_LOW_LATENCY=0
export VLLM_ASCEND_ZBAL_MOE_NVL_BYTES=10240
export VLLM_ASCEND_ZBAL_MOE_RDMA_BYTES=10240


#MODEL_PATH="/home/weights/Qwen3-32B-W8A8/"
MODEL_PATH="/home/h00932613/models/Qwen/Qwen3-30B-A3B"

SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=4
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
#export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15

export ASCEND_LAUNCH_BLOCKING=1
export MMC_LOCAL_CONFIG_PATH=/home/p00801009/vllm-ascend/vllm_test/mmc-local.conf

if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="30001"
    LOOKUP_RPC_PORT="0"
else
    KV_ROLE="kv_consumer"
    KV_PORT="30002"
    LOOKUP_RPC_PORT="1"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

if [ "$HARDWARE_SERIES" == "A2" ]; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME

elif [ "$HARDWARE_SERIES" == "A3" ]; then
    export ACL_OP_INIT_MODE=1
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2' or 'A3'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset PYTORCH_NPU_ALLOC_CONF
export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True

KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "'$KV_ROLE'",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "'$KV_ROLE'",
        "kv_port": "'$KV_PORT'",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          },
          "decode": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          }
        }
      }
    ]
  }
}'

KV_CONFIG_OLD='{
  "kv_connector": "MultiConnector",
  "kv_role": "'$KV_ROLE'",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "'$KV_ROLE'",
        "kv_port": "'$KV_PORT'",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          },
          "decode": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "'$KV_ROLE'",
        "kv_connector_extra_config": {
          "backend": "memcache",
          "lookup_rpc_port": "'$LOOKUP_RPC_PORT'"
        }
      }
    ]
  }
}'


CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port 40060
  --max-num_seqs 40
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/p00801009/vllm-ascend/vllm_test/vllm_profile", "torch_profiler_with_stack": false}'
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" 2>&1 | tee log_${ROLE}.log

echo "vLLM started. Log file: log_${ROLE}.log"
