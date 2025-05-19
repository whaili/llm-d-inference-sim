#!/bin/bash

# Copyright 2025 The llm-d-inference-sim Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This shell script launches kind and runs the vllm simulator
# as a set of deployments.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

tput bold
echo "Create the vLLM cluster:"
tput sgr0
echo ""
kind create cluster --name vllm --config ${DIR}/../yaml/vllm-cluster.yaml
kind load docker-image llm-d-inference-sim/llm-d-inference-sim:0.0.2 --name vllm

echo ""
tput bold
echo "Launch vLLM pods:"
tput sgr0
echo ""
kubectl apply --context kind-vllm -f ${DIR}/../yaml/vllm-kind.yaml
