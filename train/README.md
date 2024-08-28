## 🖥️ LongWriter Training

### Data preprocessing

First, tokenize the raw text data using the tokenizer of the model. Run the code `pre_tokenize_glm4.py` for GLM-4-9B or `pre_tokenize_llama3.py` for Llama-3.1-8B. Remember to add your general SFT data path. please format your data as follows: 
```json
{
    "messages": [{"role": "user", "content": "..."}, 
                 {"role": "assistant", "content": "..."}, ...]
    }
```

We use [packing](https://arxiv.org/abs/2401.18058) strategy for more efficient training, run
```bash
python sort_and_group.py --train_file ./data/glm4/longwriter
```
to organize the tokenized data for packing training.

### Model training

We provide training scripts under `scripts/` for the GLM-4-9B and Llama-3.1-8B model series. Make sure to adjust `--model_name_or_path`, `--train_file`, and `--output_dir` to match your model path, data path, and output path.

To support packing training, we provide patch files under `patch/`, please replace the original modeling files with them.

**Environment**: `transformers==4.33.0` for `GLM-4-9B` and `transformers==4.43.0` for `Llama-3.1-8B`.
