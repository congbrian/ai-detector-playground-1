import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleDetector:
    def __init__(self, model1_name="gpt2", model2_name="distilgpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # model 1
        self.model1 = AutoModelForCausalLM.from_pretrained(model1_name).to(self.device)
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
        
        # model 2
        self.model2 = AutoModelForCausalLM.from_pretrained(model2_name).to(self.device)
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
        
        # Perplexity ratio
        self.threshold = 1.1

    def compute_perplexity(self, model, tokenizer, text):
        # tokenizer
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        # ignore gradient descent
        with torch.no_grad():
            outputs = model(**inputs)
        #get logits
        logits = outputs.logits[:, :-1].contiguous()
        # shift labels by 1 so we get predictions
        labels = inputs.input_ids[:, 1:].contiguous()

        # how confused are we
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean')
        perplexity = torch.exp(loss).item()
        return perplexity

    def compute_score(self, text):
        perplexity1 = self.compute_perplexity(self.model1, self.tokenizer1, text)
        perplexity2 = self.compute_perplexity(self.model2, self.tokenizer2, text)
        # this isn't really cross perplexity but it's something
        return perplexity1 / perplexity2

    def predict(self, text):
        try:
            score = self.compute_score(text)
            result = "Most likely AI-generated" if score < self.threshold else "Most likely human-generated"
            return f"Perplexity ratio: {score:.2f}, {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Usage example
detector = SimpleDetector()

sample_texts = [
    "I am a paste salesman, would you like to buy paste?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is revolutionizing various industries and transforming the way we live and work.",
    "In a groundbreaking study, researchers have discovered a new species of deep-sea creature that defies current understanding of marine biology.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    "Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures."
]

for text in sample_texts:
    print(f"\nText: {text}")
    print(f"Prediction: {detector.predict(text)}")