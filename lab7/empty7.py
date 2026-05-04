from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["I love AI", "I hate bugs"]
labels = torch.tensor([1, 0])

encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

model.train()

for epoch in range(3):
    outputs = model(**encodings, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
