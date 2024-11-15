CUDA_VISIBLE_DEVICES=$1 python3 -m experiments.evaluate \
    --alg_name CONTEXT_FT \
    --model_name=../huggingface/gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --dataset_size_limit=200 \
    --num_edits=100 \
    --use_cache \
    --no_edit