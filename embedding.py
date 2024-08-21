from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
model = AutoModel.from_pretrained("klue/roberta-base")


def embed_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")

    # 모델을 사용하여 문장 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)

    sentence_embedding = outputs.last_hidden_state.mean(dim=1)

    return sentence_embedding
