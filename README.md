=
# assenbly-Alpaca

AI 국회 어드바이저 모델 개발

KoAlpaca 모델에 국회 데이터를 학습시켜 (LoRA finetuning) 법률 자문을 해줄 수 있는 언어모델을 개발한다.

![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/8f6550d7-35a3-45af-98c8-1704c530368d)
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/3078da85-5eb0-4448-9d29-5f5baf1383bc)
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/9897cbdc-f995-459d-b6a7-8f8a4dcced85)



## Pretrained model
Pretrained model link : [hyunseoki/ko-en-llama2-13b](https://huggingface.co/hyunseoki/ko-en-llama2-13b)

## Data
국회데이터

[huggingface dataset](https://huggingface.co/datasets/bong9/assemblydata)에도 올려놓았습니다.

datasets library에서 이 dataset을 바로 불러올 수 있습니다 :

```python
from datasets import load_dataset

dataset = load_dataset("bong9/assemblydata")
```

## Copyright Policy

[국회 공모전 대회 홈페이지](https://www.assembly00data.com/summary/summary.php)
"누구에게나 개방되어있으며, 영리 목적을 포함하여 모든 자유로운 활동이 보장됩니다."


## Resources

- **학습 코드 (Colab)**: 학습 코드는 [여기](https://colab.research.google.com/drive/1OjyOK1JGg10QKYjEWsHchX1CiWiEH_si?usp=sharing)에서 확인할 수 있습니다.
- **Model**: Hugging Face에도 [adapter_model](https://huggingface.co/juicyjung/ko_law_alpaca-12.8b) 파일 업로드 해놓았습니다.


## Usage

PEFT library에서 이 모델을 불러올 수 있습니다 :

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_id = "Bong9/easydata"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()
```

법률안 질문에 대한 답변을 생성하기 위해 다음과 같은 코드를 사용합니다 :

```python
def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=50,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))


gen('보건의료에 관심이 많은데 어떤 국회의원에게 관심을 가져야할까요?')
```


"# assembly-alphaca" 
