# POEF - Policy Effective Jailbreak Attacks

POEF is a red-teaming framework for evaluating the security of LLM-based robots against jailbreak attacks. The framework implements an automated attack methodology that leverages the hidden-layer gradients from unaligned LLM to guide victim LLMs to generate effective policies.

### Methodology

The framework uses a three-stage approach:
1. **Mutation**: Generates mutant samples using unaligned token gradients
2. **Selection**: Selects optimal prompts based on hidden layer loss between aligned and unaligned models
3. **Evaluation**: Uses multi-agent judgment to assess attack success

## Installation

### Requirements
- Python >= 3.9
- CUDA-capable GPU (recommended)

### Install from source

```bash
git clone <repository-url>
cd POEF
pip install -e .
pip install -r requirements.txt
```

## Usage


```bash
python src/run_poef.py \
  --task_name "Qwen2.5-7B-Instruct" \
  --dataset_path datasets/harmful_rlbench.jsonl \
  --attack_model_path Qwen/Qwen2.5-7B-Instruct \
  --attack_model_name qwen-7b-chat \
  --target_model_path Qwen/Qwen2.5-7B-Instruct \
  --target_model_name qwen-7b-chat \
  --jailbreak_prompt_length 10 \
  --num_turb_sample 64 \
  --batchsize 16 \
  --max_num_iter 20
```


## Project Structure

```
POEF/
├── poef/                      # Main package
│   ├── attacker/             # Attack implementations
│   │   ├── POEF.py          # Main POEF attacker
│   │   └── attacker_base.py # Base attacker class
│   ├── models/              # Model wrappers
│   │   ├── huggingface_model.py  # HuggingFace model interface
│   │   ├── openai_model.py       # OpenAI API interface
│   │   └── model_base.py         # Base model class
│   ├── mutation/            # Mutation strategies
│   │   └── unaligned_token_gradient.py
│   ├── selector/            # Selection policies
│   │   └── HiddenLayerLossSelector.py
│   ├── metrics/             # Evaluation metrics
│   │   ├── Evaluator.py
│   │   └── Evaluator_MultiAgentJudge.py
│   ├── datasets/            # Dataset handling
│   │   ├── jailbreak_datasets.py
│   │   └── instance.py
│   ├── seed/                # Seed generation
│   ├── utils/               # Utility functions
│   └── loggers/             # Logging utilities
├── src/                     # Run scripts
│   ├── run_poef.py.         # Main python file
├── datasets/                # Attack datasets
├── prompts/                 # Prompt templates
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── LICENSE                 # License file
└── README.md               # This file
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Disclaimer

This framework is designed for research and educational purposes to improve LLM-based robots' security. Please use it responsibly and only on robots you own or have permission to test.
