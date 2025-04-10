# Copyright 2025 The vLLM-Sim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for the vllm-sim project

PACKAGE_VLLM_SIM = github.com/neuralmagic/vllm-sim/cmd/vllm-sim
VLLM_SIM_NAME = vllm-sim/vllm-sim
VLLM_SIM_TAG ?= 0.0.2

.PHONY: build-vllm-sim
build-vllm-sim:
	go build -o bin/ ${PACKAGE_VLLM_SIM}

.PHONY: build-vllm-sim-linux
build-vllm-sim-linux:
	GOOS=linux GOARCH=amd64 go build -o bin/linux/ ${PACKAGE_VLLM_SIM}

.PHONY: build-vllm-sim-image
build-vllm-sim-image: build-vllm-sim-linux
	docker build --file build/vllm-sim.Dockerfile --tag ${VLLM_SIM_NAME}:${VLLM_SIM_TAG} ./bin/linux

