# Functionality Extraction Test Script

This directory contains a standalone test system for extracting functionalities from Java methods and analyzing test coverage.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for LLM inference)
- Sufficient GPU memory (depends on model size)

## Setup

### 1. Create a Virtual Environment

```bash
# Navigate to the sample_code directory
cd scripts/refactored_code/sample_code

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Required Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust CUDA version if needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio

# Install Hugging Face transformers
pip install transformers

# Install vllm (optional, for faster inference with vllm backend)
# Note: vllm requires specific CUDA versions and may have additional dependencies
pip install vllm

# Install accelerate (required by transformers)
pip install accelerate
```

### 3. Verify Installation

```bash
# Check Python version
python --version

# Verify packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import vllm; print('vllm installed')" 2>/dev/null || echo "vllm not installed (optional)"
```

## Directory Structure

```
sample_code/
├── data/                          # Input JSON files directory
│   ├── LRUMap.maxSize.__test_methods.json
│   └── CompositeMap.entrySet.__test_methods.json
├── output/                        # Output directory (created automatically)
├── llm_client.py                 # LLM client with batch processing support
├── test_functionality_extraction.py  # Main processing script
└── README.md                      # This file
```

## Usage

### Basic Usage

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run with default settings
python test_functionality_extraction.py
```

### Command-Line Arguments

```bash
python test_functionality_extraction.py [OPTIONS]
```

**Available Options:**

- `--model-name PATH`: Path to the model (default: `/scratch/pppaul/cache_llm_models/Qwen14B`)
  ```bash
  python test_functionality_extraction.py --model-name /path/to/your/model
  ```

- `--use-vllm`: Use vllm backend instead of AutoModelForCausalLM (faster, requires vllm installed)
  ```bash
  python test_functionality_extraction.py --use-vllm
  ```

- `--batch-size N`: Batch size for LLM processing (default: 2)
  ```bash
  python test_functionality_extraction.py --batch-size 5
  ```

- `--enable-thinking`: Enable thinking mode (default: True)
  ```bash
  python test_functionality_extraction.py --enable-thinking
  ```

- `--data-dir DIR`: Directory containing input JSON files (default: `data`)
  ```bash
  python test_functionality_extraction.py --data-dir /path/to/data
  ```

- `--output-dir DIR`: Directory for output JSON files (default: `output`)
  ```bash
  python test_functionality_extraction.py --output-dir /path/to/output
  ```

### Example Commands

```bash
# Run with vllm backend and batch size 4
python test_functionality_extraction.py --use-vllm --batch-size 4

# Run with custom model path and output directory
python test_functionality_extraction.py --model-name /path/to/model --output-dir results

# Run with all custom settings
python test_functionality_extraction.py \
    --model-name /path/to/model \
    --use-vllm \
    --batch-size 5 \
    --data-dir data \
    --output-dir output
```

## Input Format

The script expects JSON files in the `data/` directory with the following structure:

```json
{
  "candidateMethod": "org.example.ClassName.methodName(paramTypes)",
  "testMethods": [
    {
      "className": "org.example.TestClass",
      "methodName": "testMethod",
      "methodSignature": "()",
      "id": "TestClass.testMethod()",
      "shortestPath": [...],
      "sourceCode": "...",
      "javadoc": "..."
    }
  ]
}
```

## Output Format

For each input JSON file, the script generates an output JSON file in the `output/` directory with the following structure:

```json
{
  "method_under_test": "methodName",
  "class_name": "ClassName",
  "functionalities": ["functionality1", "functionality2", ...],
  "not_tested_functionalities": ["untested1", "untested2", ...],
  "tested_functionalities": ["tested1", "tested2", ...],
  "called_test_methods": [
    {
      "methodName": "testMethod",
      "functionalities": ["functionality1"]
    }
  ],
  "processing_time": 123.45,
  "success": true,
  "error_message": null
}
```

## Logging

The script generates logs in two places:

1. **Console output**: Real-time progress and status messages
2. **Log file**: `test_functionality_extraction.log` in the current directory

Logs include:
- Method processing status
- Functionalities extracted
- Test methods analyzed
- Batch processing statistics
- Processing times
- Errors and warnings

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch-size 1`
   - Use CPU mode (slower): Install CPU-only PyTorch
   - Reduce model size or use quantization

2. **vllm import error**
   - vllm is optional; use AutoModelForCausalLM instead (remove `--use-vllm` flag)
   - Or install vllm: `pip install vllm`

3. **Model not found**
   - Check model path with `--model-name` argument
   - Ensure model files are accessible
   - Download model if needed

4. **No JSON files found**
   - Check `data/` directory contains `.json` files
   - Use `--data-dir` to specify custom directory

5. **Permission errors**
   - Ensure write permissions for output directory
   - Check file permissions for input JSON files

### Getting Help

Check the log file `test_functionality_extraction.log` for detailed error messages and stack traces.

## Performance Tips

1. **Use vllm for faster inference**: `--use-vllm` flag significantly speeds up batch processing
2. **Increase batch size**: Larger batches (e.g., `--batch-size 4-8`) improve throughput if GPU memory allows
3. **GPU memory**: Monitor GPU memory usage and adjust batch size accordingly
4. **Multiple files**: The script processes files sequentially; for parallel processing, run multiple instances with different data directories

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```
