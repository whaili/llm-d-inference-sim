# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vLLM simulator written in Go that emulates vLLM HTTP endpoints without performing actual inference. It simulates OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and vLLM-specific endpoints (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/metrics`, `/health`, `/ready`, `/tokenize`).

The simulator supports two modes:
- `echo`: Returns the same text from the request
- `random`: Returns randomly chosen pre-defined sentences

Response timing is controlled by configurable latency parameters (`time-to-first-token`, `inter-token-latency`, `kv-cache-transfer-latency`).

## Development Commands

### Setup and Dependencies
```bash
# Install ZMQ dependencies (required for KV cache simulation)
make download-zmq

# Download HuggingFace tokenizer bindings
make download-tokenizer
```

### Building
```bash
# Build the binary (creates bin/llm-d-inference-sim)
make build

# Build Docker image (default tag: ghcr.io/llm-d/llm-d-inference-sim:dev)
make image-build

# On macOS, specify TARGETOS for Docker builds
make image-build TARGETOS=linux
```

### Testing
```bash
# Run all tests with Ginkgo
make test

# Run a specific test using focus
make test GINKGO_FOCUS="should handle chat completions"
```

### Code Quality
```bash
# Format code
make format

# Run linters (uses golangci-lint)
make lint
```

### Running Locally
```bash
# Run standalone
./bin/llm-d-inference-sim --model my_model --port 8000

# Using Docker
docker run --rm --publish 8000:8000 ghcr.io/llm-d/llm-d-inference-sim:dev --port 8000 --model "Qwen/Qwen2.5-1.5B-Instruct"

# Using config file
./bin/llm-d-inference-sim --config manifests/config.yaml
```

## Architecture

### Package Structure

- **`cmd/llm-d-inference-sim/`**: Main entry point that sets up logger, signal handlers, and starts the simulator
- **`pkg/llm-d-inference-sim/`**: Core simulator implementation
  - `server.go`: HTTP server setup and routing (uses fasthttprouter)
  - `simulator.go`: Request handling and response generation logic
  - `streaming.go`: Server-sent events (SSE) streaming support
  - `latencies.go`: Timing simulation logic (TTFT, inter-token latency)
  - `lora.go`: LoRA adapter management
  - `metrics.go`: Prometheus metrics implementation
  - `failures.go`: Failure injection support
- **`pkg/openai-server-api/`**: OpenAI API request/response structures and tool validation
- **`pkg/vllm-api/`**: vLLM-specific API structures (tokenization, models)
- **`pkg/kv-cache/`**: KV cache simulation with ZMQ event publishing
- **`pkg/dataset/`**: SQLite-based dataset support for generating responses from conversation history
- **`pkg/common/`**: Shared utilities

### Key Dependencies

- **fasthttp**: High-performance HTTP server framework
- **fasthttprouter**: HTTP routing
- **Ginkgo/Gomega**: BDD-style testing framework
- **klog**: Kubernetes-style logging
- **prometheus/client_golang**: Metrics exposure
- **pebbe/zmq4**: ZeroMQ bindings for KV cache events
- **mattn/go-sqlite3**: Dataset storage (CGO dependency)
- **daulet/tokenizers**: HuggingFace tokenizer bindings (CGO dependency)

### CGO Requirements

This project requires CGO for:
1. **Tokenizers**: HuggingFace tokenizer bindings (`libtokenizers.a`)
2. **ZMQ**: ZeroMQ library for KV cache event publishing
3. **SQLite**: Database support for dataset-based responses

The Makefile handles downloading tokenizer bindings. ZMQ must be installed system-wide (use `make download-zmq`).

### Configuration

The simulator accepts configuration via:
1. Command-line flags (defined using `spf13/pflag`)
2. YAML config file (path specified with `--config`)
3. Environment variables (`POD_NAME`, `POD_NAMESPACE`)

Command-line flags override config file values. See `manifests/config.yaml` for an example configuration.

### Data Parallel Support

The simulator supports Data Parallel deployment (`--data-parallel-size` from 1-8). Rank 0 runs on the configured `--port`, subsequent ranks on port+1, port+2, etc.

### Testing Strategy

- Tests use Ginkgo BDD framework
- Each major package has a test suite (`*_suite_test.go`)
- Test fixtures and helpers are in `*_fixture_test.go` and `test_helpers.go`
- golangci-lint configuration in `.golangci.yml` includes Ginkgo-specific linters

### TLS Support

TLS can be enabled via configuration. See `pkg/llm-d-inference-sim/server_tls.go` for SSL configuration logic.

## Reply Guidelines
- Always reference **file path + function name** when explaining code.
- Use **Mermaid diagrams** for flows, call chains, and module dependencies.
- If context is missing, ask explicitly which files to `/add`.
- Never hallucinate non-existing functions or files.
- Always reply in **Chinese**

## Excluded Paths
- vendor/
- build/
- dist/
- .git/
- third_party/

## Glossary


## Run Instructions
