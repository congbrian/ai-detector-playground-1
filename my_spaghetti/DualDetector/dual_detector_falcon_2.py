import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
import numpy as np

class DualDetector_2:

    # the majority of the code is from binoculars but slightly adjusted
    # particularly the __init__ is more custom
    def __init__(self, 
                 observer_name: str = "tiiuae/falcon-7b",
                 performer_name: str = "tiiuae/falcon-7b-instruct",
                 # interesting to see if we can go lower, bino uses 16.
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 temperature: float = 1.0,
                 median: bool = False,
                 sample_p: bool = False,
                 device: str = None):
        
        # generic construction
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = use_bfloat16
        self.observer_model = self._load_model(observer_name)
        self.performer_model = self._load_model(performer_name)

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # passed to logic
        self.max_token_observed = max_token_observed
        self.temperature = temperature
        self.median = median
        self.sample_p = sample_p

        # ce_loss_fn and soft_max are bino
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax_fn = torch.nn.Softmax(dim=-1)

        self.change_mode(mode)


    def _load_model(self, model_name: str):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16 if self.use_bfloat16 else torch.float32
        ).eval()

    # just preserving this since it's included with bino.
    def change_mode(self, mode: str):
        if mode == "low-fpr":
            self.threshold = 0.8536432310785527
        elif mode == "accuracy":
            self.threshold = 0.9015310749276843
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # had to remove some stuff to get it to run
    def _tokenize(self, batch: list[str]):
        batch_size = len(batch)
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed
        ).to(self.device)

    @torch.inference_mode()
    def _get_logits(self, encodings):
        # from bino metrics
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        return observer_logits, performer_logits

    def perplexity(self, encoding, logits):
        # from bino metrics, exclude median and temp as included in self.
        shifted_logits = logits[..., :-1, :].contiguous() / self.temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        if self.median:
            ce_nan = (self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                      masked_fill(~shifted_attention_mask.bool(), float("nan")))
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
        else:
            ppl = (self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                   shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        return ppl

    def entropy(self, p_logits, q_logits, encoding):
        # from bino metrics, excludes pad token, median, sample p, and temp as included in self.
        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores, q_scores = p_logits / self.temperature, q_logits / self.temperature

        p_proba = self.softmax_fn(p_scores).view(-1, vocab_size)

        if self.sample_p:
            p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

        q_scores = q_scores.view(-1, vocab_size)

        ce = self.ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (encoding.input_ids != self.tokenizer.pad_token_id).type(torch.uint8)

        if self.median:
            ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
            agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
        else:
            agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

        return agg_ce

    # removed calls to DEVICE_1
    def compute_score(self, input_text: Union[str, list[str]]) -> list[float]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = self.perplexity(encodings, performer_logits)
        x_ppl = self.entropy(observer_logits, performer_logits, encodings)

        binoculars_scores = (ppl / x_ppl).tolist()  # Convert to list here
        return binoculars_scores if isinstance(input_text, list) else [binoculars_scores[0]]
    
    def predict(self, input_text: Union[str, list[str]]) -> Union[str, list[str]]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred


# Result
# detector = DualDetector_2(use_bfloat16=True)

# sample_texts = [
#     "I am a paste salesman, would you like to buy paste?",
#     "The quick brown fox jumps over the lazy dog.",
#     "Artificial intelligence is revolutionizing various industries and transforming the way we live and work.",
#     "In a groundbreaking study, researchers have discovered a new species of deep-sea creature that defies current understanding of marine biology.",
#     "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
#     "Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.",
#     "World of Warcraft (often abbreviated as WoW) is a massively multiplayer online role-playing game (MMORPG) by Blizzard Entertainment. This subreddit is dedicated to the discussion of things in and around the game. We are not directly affiliated with Blizzard (though we have a friendly relationship with them!) and we are not a replacement for the official forums, but instead a way to interact with your favourite game while on your favourite website."
#     "extensive amount of data generated by today's clinical systems, has led to the development of imaging AI solutions across the whole value chain of medical imaging, including image reconstruction, medical image segmentation, image-based diagnosis and treatment planning. Notwithstanding the successes and future potential of AI in medical imaging, many stakeholders are concerned of the potential risks and ethical implications of imaging AI solutions, which are perceived as complex, opaque, and difficult to comprehend, utilise, and trust in critical clinical applications. Despite these concerns and risks, there are currently no concrete guidelines and best practices for guiding future AI developments in medical imaging towards increased trust, safety and adoption. To bridge this gap, this paper introduces a careful selection of guiding principles drawn from the accumulated experiences, consensus, and best practices from five large European projects on AI in Health Imaging. These guiding principles are named FUTURE-AI and its building blocks consist of (i) Fairness, (ii) Universality, (iii) Traceability, (iv) Usability, (v) Robustness and (vi) Explainability. In a step-by-step approach, these guidelines are further translated into a framework of concrete recommendations for specifying, developing, evaluating, and deploying technically, clinically and ethically trustworthy AI solutions into clinical practice."
# ]

# for text in sample_texts:
#     print(f"\nText: {text}")
#     print(f"Score: {detector.compute_score(text):.4f}")
#     print(f"Prediction: {detector.predict(text)}")