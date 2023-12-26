# Chat-Doctor
Finetuning Mistral-7B into a Medical Chat Doctor using Huggingface 🤗+ QLoRA + PEFT.

Will upload the trained weights in few days, thank you!


## Introduction
**Hi there!
**
_Have you ever wondered what’s it like to finetune a large language model (LLM) on your own custom dataset? Well there are some resources which can help you to achieve that, but frankly speaking even after reading those heavy ML infused articles and notebooks one can’t just train LLMs straightaway on your home pc or laptops unless it has some decent GPUs!

I recently got access to a Sagemaker ml.g4dn.12xlarge instance, which provides you 4 Nvidia T4 GPUs (64GB VRAM) and I am sharing my experience of how I finetuned Mistral-7B into a Chat Doctor!
_
## The dataset
I have used a dataset called ChatDoctor-HealthCareMagic-100k, it contains about 110K+ rows of patient’s queries and the doctor’s opinions.

I didn’t use the entirety of this dataset though, for my experimentation I used around 5000 randomly sampled rows from this dataset.

So this is what the dataset looks like, 90% is being used for train and 10% is being used for eval.

DatasetDict({
    train: Dataset({
        features: ['instruction', 'input', 'output'],
        num_rows: 4500
    })
    test: Dataset({
        features: ['instruction', 'input', 'output'],
        num_rows: 500
    })
})
