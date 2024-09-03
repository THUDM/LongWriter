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

### FAQ
1. Error when running training script: ⚠️`DeepSpeedZeroConfig
stage3_prefetch_bucket_size Input should be a valid integer, got a number with a fractional part`. **This may happen if your `deepspeed>=0.15.0`, we suggest downgrade to `deepspeed==0.14.4` to resolve this issue.**
2. Error when training GLM-4-9b: ⚠️`return self.mergeable_ranks[token] KeyError: '<|endoftext|>'`. **Please make sure you have replaced the `tokenization_chatglm.py` and `modeling_chatglm.py` files under the GLM-4-9b model folder by the patch files under `patch/`. Also make sure your environment satisfy the requirements: `transformers==4.33.0` and `flash-attn>=2.0.0`.**
3. Encountered during GLM-4-9b training: ⚠️`RuntimeError: shape '[32768, -1, 1, 32, 2]' is invalid for...`. **Please modify the `"seq_length"` in `config.json` of GLM-4-9b from 8192 to 131072.**
