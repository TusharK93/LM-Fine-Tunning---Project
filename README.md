# LLM Fine-Tuning

A hands-on collection of notebooks demonstrating different approaches to fine-tuning Large Language Models (LLMs), including **LoRA, QLoRA, and PEFT**. The project explores model adaptation using instruction-response datasets and compares model behavior before and after fine-tuning.

## Project Overview

Fine-tuning adapts a pre-trained language model to a particular task, domain, or response style by training it on a smaller, task-specific dataset. This repository contains notebook-based experiments with Llama 2, Gemma, Mistral-related configuration, and Gradient AI.

The notebooks cover:
- Preparing instruction-response examples for supervised fine-tuning.
- Loading pre-trained language models and datasets.
- Applying parameter-efficient fine-tuning techniques.
- Configuring training parameters such as learning rate, batch size, epochs, and LoRA rank.
- Comparing model responses before and after fine-tuning.
- Saving or publishing fine-tuned model weights where demonstrated.

## Repository Structure

```text
Finetuning-LLM/
├── Fine_Tuning_LLm_Models.ipynb
├── Fine_Tuning_with_Mistral_QLora_PEFt.ipynb
├── Fine_tune_Llama_2.ipynb
├── lora_tuning.ipynb
├── LICENSE
└── README.md
```

## Notebooks

### 1. Fine-tuning with Gradient AI

**File:** `Fine_Tuning_LLm_Models.ipynb`

This notebook demonstrates adapter-based fine-tuning through the Gradient AI Python SDK.

Main steps:
1. Install the `gradientai` package.
2. Connect to the Gradient service.
3. Retrieve the `nous-hermes2` base model.
4. Create a model adapter.
5. Test the base adapter with an instruction prompt.
6. Fine-tune using a small collection of instruction-response examples.
7. Compare the generated response before and after fine-tuning.

The example dataset contains instruction-response pairs about a specific subject. It illustrates how examples can guide a model toward a desired response.

### 2. Fine-tuning Mistral with QLoRA and PEFT

**File:** `Fine_Tuning_with_Mistral_QLora_PEFt.ipynb`

This notebook introduces a configuration-based workflow for fine-tuning with Mistral, QLoRA, and PEFT.

The notebook includes:
- A setup section for installing libraries.
- A configuration dictionary containing a model name, QLoRA and PEFT flags, learning rate, and number of epochs.
- A model initialization step.
- Sections for data loading, preprocessing, fine-tuning, and evaluation.

**Note:** The data preparation and evaluation sections are placeholders, and the notebook should be treated as an introductory outline rather than a complete, verified training pipeline. The configured model name is `bert-base-uncased`; verify model/library compatibility and the actual implementation before running this notebook as a Mistral fine-tuning workflow.

### 3. Fine-tuning Llama 2 with QLoRA

**File:** `Fine_tune_Llama_2.ipynb`

This notebook demonstrates instruction fine-tuning of the Llama 2 7B Chat model using QLoRA.

Key components:
- **Base model:** `NousResearch/Llama-2-7b-chat-hf`
- **Dataset:** `mlabonne/guanaco-llama2-1k`
- **Fine-tuning approach:** QLoRA with PEFT
- **Quantization:** 4-bit quantization using the NF4 configuration
- **LoRA rank:** 64
- **LoRA alpha:** 16
- **LoRA dropout:** 0.1
- **Training duration:** 1 epoch in the demonstrated configuration

The workflow loads the dataset and tokenizer, configures 4-bit quantization, prepares the model for parameter-efficient training, and uses the supervised fine-tuning trainer. It then saves the adapter, demonstrates text generation, and shows how to merge LoRA weights with the base model and publish the resulting model and tokenizer to the Hugging Face Hub.

The notebook also demonstrates TensorBoard logging for monitoring training.

### 4. Fine-tuning Gemma with LoRA

**File:** `lora_tuning.ipynb`

This notebook follows Google's Gemma fine-tuning tutorial and uses Keras 3 with KerasNLP.

Key components:
- **Base model:** Gemma 2B (`gemma_2b_en`)
- **Dataset:** Databricks Dolly 15K
- **Training subset:** 1,000 examples in the demonstration
- **Fine-tuning approach:** LoRA
- **LoRA rank:** 4
- **Sequence length:** 512
- **Optimizer:** AdamW
- **Learning rate:** `5e-5`
- **Training duration:** 1 epoch
- **Backend configured:** JAX

The notebook formats instruction-response examples, checks model responses before training, enables LoRA, fine-tunes the model, and tests the responses again. The before-and-after examples illustrate how fine-tuning can adapt outputs to follow instructions more closely.

## Fine-Tuning Concepts

| Technique | Description |
|---|---|
| Fine-tuning | Further training a pre-trained model on task-specific examples. |
| PEFT | Parameter-Efficient Fine-Tuning methods that reduce the number of parameters updated during training. |
| LoRA | Freezes the original model weights and trains smaller low-rank adapter matrices. |
| QLoRA | Combines quantization with LoRA to reduce memory requirements during fine-tuning. |
| 4-bit quantization | Represents model weights with lower precision to reduce memory usage. |
| Supervised fine-tuning (SFT) | Trains a model on examples containing prompts or instructions and desired responses. |

## General Workflow

```text
Pre-trained LLM
      |
      v
Select and prepare dataset
      |
      v
Format instruction-response examples
      |
      v
Configure model and tokenizer
      |
      v
Choose fine-tuning method
(LoRA / QLoRA / PEFT)
      |
      v
Train the model
      |
      v
Evaluate generated responses
      |
      v
Save or publish model
      |
      v
Run inference with the fine-tuned model
```

## Technologies and Libraries

The notebooks use different tools depending on the experiment:

- Python
- Jupyter Notebook / Google Colab
- Hugging Face Transformers and Datasets
- PEFT
- bitsandbytes
- TRL / supervised fine-tuning utilities (in the Llama 2 workflow)
- PyTorch
- Keras 3 and KerasNLP
- TensorBoard
- Gradient AI SDK
- Hugging Face Hub

Install the dependencies required by the specific notebook you want to run. The notebooks do not share one common, pinned environment.

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Finetuning-LLM
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install notebook-specific dependencies

Install the packages listed in the notebook you intend to run. Some experiments require a compatible GPU, additional model access, or external service credentials.

### 4. Run a notebook

```bash
jupyter notebook
```

Open the notebook you want to explore and execute its cells in order. Google Colab can also be used for notebooks that support it.

## Hardware and Access Notes

- LLM fine-tuning can require substantial GPU memory and compute.
- The Llama 2 QLoRA notebook is designed to reduce memory usage through 4-bit quantization; actual requirements depend on the runtime, sequence length, batch settings, and model implementation.
- The Gemma notebook describes using a Google Colab T4 GPU for the Gemma 2B workflow.
- Some models or datasets may require accepting access terms or authenticating with their hosting platform.
- Gradient AI experiments require valid Gradient credentials and service access.
- Check the current model licenses, dataset terms, and platform requirements before using the artifacts commercially.

## Important Security Note

Never commit API keys, access tokens, passwords, or other credentials to GitHub. Use environment variables or notebook secret storage instead. If a credential has already been committed or exposed, revoke or rotate it and remove it from the repository's history as appropriate. Do not reuse the credential.

## Limitations

- These are separate learning experiments, not a unified production fine-tuning framework.
- The Mistral notebook contains placeholder data-preparation and evaluation steps and requires further implementation.
- The Gradient AI notebook uses a small illustrative dataset; its results should not be treated as a comprehensive model evaluation.
- The Llama 2 and Gemma notebooks use limited training configurations for demonstration. Better results require task-appropriate data, systematic evaluation, and careful hyperparameter tuning.
- No benchmark results or quantitative performance improvements are reported consistently across all notebooks.

## Future Improvements

- Add reusable dataset preparation and validation scripts.
- Add train, validation, and test splits.
- Evaluate models using task-specific metrics and human review.
- Compare base and fine-tuned models on the same held-out prompts.
- Track experiments, hyperparameters, and evaluation results.
- Add inference scripts and an API for serving fine-tuned models.
- Pin dependency versions and document tested hardware configurations.
- Add data and model cards describing provenance, intended use, and limitations.

## License

See the [`LICENSE`](LICENSE) file for the repository's license. Check the licenses and usage terms of each base model, dataset, and external service separately.

---

**Purpose:** Educational experiments for understanding and exploring LLM fine-tuning techniques, especially LoRA, QLoRA, and PEFT.
