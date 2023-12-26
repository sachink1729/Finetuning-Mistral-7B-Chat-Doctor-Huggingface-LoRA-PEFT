# Chat-Doctor
Finetuning Mistral-7B into a Medical Chat Doctor using Huggingface 🤗+ QLoRA + PEFT.

Will upload the trained weights in few days, thank you!


## 1. Introduction

Hi there!
Have you ever wondered what’s it like to finetune a large language model (LLM) on your own custom dataset? Well there are some resources which can help you to achieve that, but frankly speaking even after reading those heavy ML infused articles and notebooks one can’t just train LLMs straightaway on your home pc or laptops unless it has some decent GPUs!

I recently got access to a Sagemaker ml.g4dn.12xlarge instance, which provides you 4 Nvidia T4 GPUs (64GB VRAM) and I am sharing my experience of how I finetuned Mistral-7B into a Chat Doctor!

## 2. The dataset
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

## 3. Formatting Prompts and Tokenizing the dataset.
If you see the structure of this data it contains instruction, input and an output column, what we need to do is to format it in a certain manner in order to feed it into a LLM.

Since LLMs are basically fancy Transformer decoders, you just combine everything in a certain format, tokenize it and give it to the model.

To format the rows we can use a very basic format function. Here inputs are patient queries and outputs are doctor’s answers.

Example:

### The following is a doctor's opinion on a person's query: 
### Patient query: I have considerable lower back pain, also numbness in left buttocks and down left leg, girdling at the upper thigh.  MRI shows \"Small protrusiton of L3-4 interv. disc on left far laterally with annular fissuring fesulting in mild left neural foraminal narrowing with slight posterolateral displacement of the exiting l3 nerve root.\"  Other mild bulges L4-5 w/ fissuring, and mild buldge  L5-S1. 1) does this explain symptoms 2) I have a plane/car trip in 2 days lasting 8 hrs, then other travel.  Will this be harmful? 
### Doctor opinion: Hi, Your MRI report does explain your symptoms. Travelling is possible providing you take certain measures to make your journey as comfortable as possible. I suggest you ensure you take adequate supplies of your painkillers. When on the plane take every opportunity to move about the cabin to avoid sitting in the same position for too long. Likewise, when travelling by car, try to make frequent stops, so you can take a short walk to move your legs.  Chat Doctor.

We can tokenize the dataset using Huggingface’s AutoTokenizer, point to be noted is I have used the padding token as eos_token which is end of sequence token and padding style is left so basically padding tokens will be added at the beginning of the sequence rather than in the end, why do we do that is because in decoder only architecture the output is a continuation of the input prompt — there would be gaps in the output without left padding.

The model we are using is Mistral AI’s 7B model mistralai/Mistral-7B-v0.1 https://huggingface.co/mistralai/Mistral-7B-v0.1.

## 4. Initializing Mistral-7B using Quantized low rank adaptation (QLoRA).
You can look at https://huggingface.co/blog/4bit-transformers-bitsandbytes to learn more about 4-bit quantization.
But basically what we are doing is quantizing and loading the model in 4-bit mode that drastically reduces the memory footprint of the model without changing the performance that much!
---------------------------------------------------------
## 5. PEFT and LoRA config
You might wonder what is PEFT? PEFT is Parameter Efficient Fine-tuning, its a technique that allows us to freeze most of the model params and tries to train a small percentage of the model params it supports low data scenarios to efficiently finetune the LLM on your domain dataset.

To understand more about PEFT read this blog on Huggingface https://huggingface.co/blog/peft.

Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. For that use the prepare_model_for_kbit_training method from PEFT.

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

## 6. Training 
The train took me around 15–20 hours, i don’t remember the exact time since i was resuming the train from checkpoints because of Out of memory errors in jupyter notebooks.

But I could see considerable changes in response since the first save itself at 25 steps, the responses kept getting better at each save inferences.

One important point to be noted is these saves are not the model weights but rather the QLoRA adaptor saved weights, you have to combine these weights with the original model to generate responses!

I trained it for 500 steps and here are the results using that checkpoint.


## 7. Inference using Gradio
![image](https://github.com/sachink1729/Chat-Doctor/assets/58906183/afbf8c2b-a348-478b-80eb-6e6f0c350089)

![image](https://github.com/sachink1729/Chat-Doctor/assets/58906183/8b9031b0-6774-4187-9897-08147ff27904)
