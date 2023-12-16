# Chat-Doctor
Finetuning Mistral-7B into a Medical Chat Doctor using Huggingface 🤗+ QLoRA + PEFT.

Introduction
Hi there!

Have you ever wondered what’s it like to finetune a large language model (LLM) on your own custom dataset? Well there are some resources which can help you to achieve that, but frankly speaking even after reading those heavy ML infused articles and notebooks one can’t just train LLMs straightaway on your home pc or laptops unless it has some decent GPUs!

I recently got access to a Sagemaker ml.g4dn.12xlarge instance, which provides you 4 Nvidia T4 GPUs (64GB VRAM) and I am sharing my experience of how I finetuned Mistral-7B into a Chat Doctor!

Step 1: The dataset
I have used a dataset called ChatDoctor-HealthCareMagic-100k, it contains about 110K+ rows of patient’s queries and the doctor’s opinions.


an example from the dataset
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
Step 2: Formatting Prompts and Tokenizing the dataset.
If you see the structure of this data it contains instruction, input and an output column, what we need to do is to format it in a certain manner in order to feed it into a LLM.

Since LLMs are basically fancy Transformer decoders, you just combine everything in a certain format, tokenize it and give it to the model.

To format the rows we can use a very basic format function. Here inputs are patient queries and outputs are doctor’s answers.

def formatting_func(example):
    text = f"### The following is a doctor's opinion on a person's query: \n### Patient query: {example['input']} \n### Doctor opinion: {example['output']}"
    return text
An example of how the format looks like for train example.

### The following is a doctor's opinion on a person's query: 
### Patient query: I have considerable lower back pain, also numbness in left buttocks and down left leg, girdling at the upper thigh.  MRI shows \"Small protrusiton of L3-4 interv. disc on left far laterally with annular fissuring fesulting in mild left neural foraminal narrowing with slight posterolateral displacement of the exiting l3 nerve root.\"  Other mild bulges L4-5 w/ fissuring, and mild buldge  L5-S1. 1) does this explain symptoms 2) I have a plane/car trip in 2 days lasting 8 hrs, then other travel.  Will this be harmful? 
### Doctor opinion: Hi, Your MRI report does explain your symptoms. Travelling is possible providing you take certain measures to make your journey as comfortable as possible. I suggest you ensure you take adequate supplies of your painkillers. When on the plane take every opportunity to move about the cabin to avoid sitting in the same position for too long. Likewise, when travelling by car, try to make frequent stops, so you can take a short walk to move your legs.  Chat Doctor.
Now lets tokenize the dataset!

We can tokenize the dataset using Huggingface’s AutoTokenizer, point to be noted is I have used the padding token as eos_token which is end of sequence token and padding style is left so basically padding tokens will be added at the beginning of the sequence rather than in the end, why do we do that is because in decoder only architecture the output is a continuation of the input prompt — there would be gaps in the output without left padding.

The model we are using is Mistral AI’s 7B model mistralai/Mistral-7B-v0.1.

base_model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token
Now that we have initialized our tokenizer lets tokenize the dataset.

max_length = 512 # differs from datasets to datasets

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = dataset['train']
eval_dataset = dataset['test']
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
The max length of inputs can be found by analyzing what percentage of dataset falls into your max length, for me around 98% dataset has <512 sequence length so I fixed it at 512, do your own analysis, but please be aware high length sequences will take more time to train.

Lets see how the tokenized data looks like.

[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 774, 415, 2296, 349, 264, 511, 310, 783, 28742, 28713, 7382, 302, 264, 1338, 28742, 28713, 5709, 28747, 28705, 13, 27332, 4186, 722, 5709, 28747, 18365, 396, 16993, 3358, 2774, 8513, 567, 776, 8295, 264, 2475, 1856, 305, 1804, 325, 270, 2429, 28705, 28740, 17013, 22640, 28731, 298, 1749, 2081, 302, 1749, 16229, 28723, 1770, 16229, 3358, 28723, 28705, 7164, 264, 6273, 266, 1721, 438, 272, 396, 16993, 304, 6759, 1401, 272, 16229, 2698, 28804, 28705, 1136, 8328, 354, 264, 1832, 304, 622, 24517, 1871, 298, 3221, 6300, 28723, 28705, 13, 27332, 16505, 7382, 28747, 22557, 28808, 7812, 368, 354, 272, 5709, 28723, 1047, 736, 403, 707, 11254, 5915, 28725, 6273, 266, 442, 9353, 11254, 460, 2572, 28723, 393, 1804, 297, 456, 2698, 541, 347, 835, 264, 963, 294, 645, 1773, 262, 28725, 4012, 513, 378, 4739, 6084, 739, 6328, 28723, 20063, 1007, 963, 294, 645, 1773, 262, 541, 11634, 4242, 533, 322, 3416, 28723, 415, 3358, 541, 2847, 13713, 298, 272, 277, 2121, 297, 1259, 1222, 304, 272, 1719, 3572, 541, 347, 2169, 28723, 315, 3397, 368, 298, 7731, 264, 1147, 14535, 395, 378, 28723, 560, 1222, 302, 28687, 28725, 2664, 261, 19474, 28742, 9271, 12423, 687, 1023, 347, 2203, 28723, 12297, 456, 622, 1316, 28723, 2678, 2178, 28723, 2]
You can see the sequence begins with a lot of 2’s that is probably the EOS token for this tokenizer and then you can see token “1", that is probably BOS (beginning of sequence) token.

Step 3: Initializing Mistral-7B using Quantized low rank adaptation (QLoRA).
To initialize the model use:

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, resume_download=True)
I won’t discuss much about how quantization works since who better to tell you than the creators themselves!

You can look at https://huggingface.co/blog/4bit-transformers-bitsandbytes to learn more about 4-bit quantization.

But basically what we are doing is quantizing and loading the model in 4-bit mode that drastically reduces the memory footprint of the model without changing the performance that much!

Before moving on to training the model lets look at how the base model works on a query:

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))
which gives us this:

Patient's Query:

 Hi doc i am in the recovery after dengue, sometimes my heart feels like its rubbing with my chest and it feels very uncomfortable, what can be my problem? 

###

Dr. Suresh Reddy's Answer:

Dear Sir/Madam,

Thank you for your query. Dengue fever is a viral infection that causes flu-like symptoms such as high fever, headache, muscle pain, nausea, vomiting, swollen glands or rash. It is caused by any one of four closely related viruses transmitted by mosquitoes. The disease shares some clinical signs with other tropical diseases like malaria, typhoid and hepatitis.

The most common symptom of dengue fever is a sudden onset of high fever (104°F) accompanied by severe headaches, pain behind the eyes, muscle and joint pains, and a characteristic skin rash. Other symptoms include nausea, vomiting, swollen lymph nodes, mild bleeding from the nose or gums, bloodshot eyes, and easy bruising.

In rare cases, dengue fever may progress to dengue hemorrhagic fever (DHF), characterized by damage to small blood vessels that causes internal bleeding, low levels of blood platelets and blood plasma leakage, which leads to shock. This form of
You can see the model gives some insights on what is dengue, but fails to answer the query fully.

Step 4: PEFT and LoRA config
You might wonder what is PEFT? PEFT is Parameter Efficient Fine-tuning, its a technique that allows us to freeze most of the model params and tries to train a small percentage of the model params it supports low data scenarios to efficiently finetune the LLM on your domain dataset.

To understand more about PEFT read this blog on Huggingface https://huggingface.co/blog/peft.

Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. For that use the prepare_model_for_kbit_training method from PEFT.

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
We can actually see what percentage of params we can train using PEFT which is pretty cool ngl.

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
Let’s print the model to examine its layers, as we will apply QLoRA to all the linear layers of the model. Those layers are q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, and lm_head.

Here we define the LoRA config.

r is the rank of the low-rank matrix used in the adapters, which thus controls the number of parameters trained. A higher rank will allow for more expressivity, but there is a compute tradeoff.

alpha is the scaling factor for the learned weights. The weight matrix is scaled by alpha/r, and thus a higher value for alpha assigns more weight to the LoRA activations.

The values used in the QLoRA paper were r=64 and lora_alpha=16, and these are said to generalize well, but we will use r=32 and lora_alpha=64 so that we have more emphasis on the new fine-tuned data while also reducing computational complexity.

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
which prints:

trainable params: 85041152 || all params: 3837112320 || trainable%: 2.2162799758751914
Out of around 3.8 billion params we can train around 0.08 billion params! That’s something.

Step 5: Training the model.
Now that we are all set for training the model lets go ahead and call the trainer module.

import transformers
from datetime import datetime

project = "chat-doctor-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-4, # Want a small lr for finetuning
        #bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train(resume_from_checkpoint=True)
The train took me around 15–20 hours, i don’t remember the exact time since i was resuming the train from checkpoints because of Out of memory errors in jupyter notebooks.

But I could see considerable changes in response since the first save itself at 25 steps, the responses kept getting better at each save inferences.

One important point to be noted is these saves are not the model weights but rather the QLoRA adaptor saved weights, you have to combine these weights with the original model to generate responses!

I trained it for 500 steps and here are the results using that checkpoint.

Step 6: The results!

To inference first we need to load the base model again.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
now we can combine the base model with the trained saved adaptor weights.

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-patient-query-finetune/checkpoint-500/")
Now before generating the results lets create a very simple UI using Gradio, if you don’t know about gradio you can checkout https://www.gradio.app/

Gradio basically helps you create a simple UI endpoint for your model, since we are aiming to build a Chat Doctor lets use the chatbot module but first lets create the function that will generate the responses.

def respond(query):
    eval_prompt = """Patient's Query:\n\n {} ###\n\n""".format(query)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    output = ft_model.generate(input_ids=model_input["input_ids"].to(device),
                           attention_mask=model_input["attention_mask"], 
                           max_new_tokens=125, repetition_penalty=1.15)
    result = tokenizer.decode(output[0], skip_special_tokens=True).replace(eval_prompt, "")
    return result
Now lets create the gradio app.

import gradio as gr

def chat_doctor_response(message, history):
    return respond(message)

demo = gr.ChatInterface(chat_doctor_response)

demo.launch()
And we’re done, this opens up a UI where you can enter your query and the Chat Doctor responds back.

Example 1:

Pretty neat!


Example 2:

Look how amazing the responses seem, at least to a non-med guy like me!


Conclusion:
This was indeed a fun experiment! A huge thanks to this goldmine of a notebook https://github.com/brevdev/notebooks/blob/main/mistral-finetune.ipynb which is what I followed to experiment on the Chat Doctor dataset!

We saw how to finetune a LLM on your custom dataset using Low rank adaptation!

We saw the possibilities GenAI presents, working in healthcare domain I can tell you the market has started or in the process to incorporate LLMs into their functions so it’s the right time to play and experiment with LLMs now!

For more such blogs please motivate me by liking and following me!

Thank you! Stay curious!
