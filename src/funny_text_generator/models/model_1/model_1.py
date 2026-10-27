"""
Model_1.

Attributes
----------
    PretrainedNetwork
"""

import torch
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.funny_text_generator.models.model_1.modules import MappingEncoder
from src.funny_text_generator.models.model_1.modules import MappingDecoder
from src.funny_text_generator.models.base import BaseModel


class PretrainedNetwork(BaseModel):
    """
    Pretrained model class.

    Parameters
    ----------
        encoder_name : str
            Name of an encoder model on HuggingFace.
        decoder_name : str
            Name of a decoder model on HuggingFace.
        latent_dim : int
            Dimension of latent space of a model.
        max_length : int
            Maximum length of an output sequence.
        n_tokens: int
            Number of prefix tokens to create from encoder output and pass
            to decoder.

    """
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        latent_dim: int,
        max_length: int,
        n_tokens: int,
        device: str = "cuda",
        word_dropout_p: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(
            max_length=max_length,
            device=device,
            word_dropout_p=word_dropout_p,
        )
        self._n_tokens = n_tokens

        self._encoder = AutoModel.from_pretrained(encoder_name)
        self._decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
        self._mapping_encoder = MappingEncoder(
            d_model=self._encoder.config.hidden_size,
            latent_dim=latent_dim,
        )
        self._mapping_decoder = MappingDecoder(
            d_model=self._decoder.config.n_embd,
            latent_dim=latent_dim,
            n_tokens=n_tokens,
        )

        self._encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self._encoder_tokenizer.padding_side = "right"
        self._decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self._decoder_tokenizer.padding_side = "right"


    def setup_lora(self, lora_modules: list[str]):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save=["ln_1", "ln_2", "ln_f"],
        )
        self.decoder = get_peft_model(self.decoder, lora_config)


    def _data_collator(self, batch):
        encoder_ids = self._encoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        decoder_ids = self._decoder_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        labels = decoder_ids['input_ids'].clone()
        labels[labels == self._decoder_tokenizer.pad_token_id] = -100

        return {
            'enc_input_ids': encoder_ids['input_ids'].to(self.config.device),
            'enc_attention_mask': encoder_ids['attention_mask'].to(self.config.device),
            'dec_input_ids': self.apply_word_dropout(
                decoder_ids['input_ids'],
                unk_id = self._decoder.config.unk_token_id,
                p=self._word_dropout_p,
            ).to(self.config.device),
            'dec_attention_mask': decoder_ids['attention_mask'].to(self.device),
        }, labels.to(self.device)


    def forward(self, batch):
        (enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask), labels = self.data_collator(batch)

        encoder_output = self._encoder(
            enc_input_ids,
            attention_mask=enc_attention_mask,
            ).last_hidden_state[:, 0, :]
        mu, log_var, z = self._mapping_encoder(encoder_output)
        z_upscaled = self._mapping_decoder(z)

        decoder_embeddings = self._decoder.transformer.wte(dec_input_ids)
        decoder_embeddings = torch.cat([z_upscaled, decoder_embeddings], dim=1)

        z_mask = torch.full((dec_attention_mask.shape[0], self.n_tokens), 1, device=self.device)
        full_decoder_attention_mask = torch.cat((z_mask, dec_attention_mask), dim=1)

        decoder_output = self.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=full_decoder_attention_mask,
            labels=None,
        ).logits[:, (self.n_tokens-1):-1, :]

        return (mu, log_var, decoder_output, z), labels


    @torch.no_grad()
    def forward_inference(
        self,
        noise: torch.Tensor,
        top_p: float,
        top_k: int,
        temperature: float,
    ):
        """
        Generate text from a T ~ N(0, 1).

        Parameters
        ----------
        top_p : float
            Top p parameter for generation.
        top_k : int
            Top k parameter for generation.
        temperature : float
            Temperature parameter for generation.
            Makes softmax distrubution smoother if < 1 and has an inverse effect if > 1.

        Returns
        -------
        generated_ids : torch.Tensor
            Output sequence.
        """
        z_upscaled = self._mapping_decoder(noise)
        generated_ids = self._decoder.generate(
            inputs_embeds=z_upscaled,
            max_new_tokens=self._max_length,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            eos_token_id=self._decoder.config.eos_token_id,
            pad_token_id=self._decoder.config.pad_token_id,
            use_cache=True,
        )
        return generated_ids
