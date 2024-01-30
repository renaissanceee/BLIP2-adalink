"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_linear=True,
        freeze_qformer=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        rank=16,
        use_adalink_I=True,
        use_adalink_T=True,
        use_adalink_qformer=True,
        

    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.wandb_name=""
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.use_adalink_qformer = False
        self.adalink_qformer = None
        # adalink
        if use_adalink_qformer==True:
            self.wandb_name="ada_qformer_"+str(rank)
            self.rank=rank  # 4,16,64,256
            self.use_adalink_qformer = True 
            logging.info("ada_qformer rank= {}".format(self.rank))
            # print(self.visual_encoder.num_features)#1408
            self.adalink_qformer=nn.Sequential(
                nn.Linear(768,self.rank),# 768->4
                nn.Linear(self.rank, 768)# 4->768
            )
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, freeze_qformer
        )
        if not freeze_qformer:
            self.wandb_name="ft_qformer"

        self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.float()
        # linear_proj for qformer: frozen
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )
        if freeze_linear:
            for param in self.t5_proj.parameters():
                param.requires_grad = False
        self.use_adalink_I, self.use_adalink_T = False, False
        self.adalink_I, self.adalink_T = None, None
        # adalink
        if use_adalink_I==True:
            self.wandb_name="adalink_"+str(rank)
            self.rank=rank  # 4,16,64,256
            self.use_adalink_I = use_adalink_I #True
            self.adalink_I=nn.Sequential(
                nn.Linear(self.t5_model.config.hidden_size,self.rank),# 2048->4
                nn.Linear(self.rank, self.t5_model.config.hidden_size)# 4->2048
            )            
            logging.info("adalink rank= {}".format(self.rank))
        if use_adalink_T==True:
            self.wandb_name="adalink_"+str(rank)
            self.rank=rank
            self.use_adalink_T = use_adalink_T #True
            self.adalink_T=nn.Sequential(
                nn.Linear(self.t5_model.config.hidden_size,self.rank),# 2048->4
                nn.Linear(self.rank, self.t5_model.config.hidden_size)# 4->2048
            )
        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            if self.use_adalink_qformer==True:
                print('image_embeds:',image_embeds.size())
                image_embeds=image_embeds+self.adalink_qformer(image_embeds)#[196, 768]
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)#[bz,32,2048]
        if self.use_adalink_I:
            inputs_t5 = inputs_t5 + self.adalink_I(inputs_t5)#[1, 32, 2048]->2048->[1, 32, 2048]  
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
        # with torch.cuda.amp.autocast():
            # print(samples.keys())
            # vqav2 ['image', 'text_input', 'answer', 'weight', 'n_answers', 'epoch', 'num_iters_per_epoch', 'iters']
            # coco_caption ['image', 'text_input', 'image_id', 'epoch', 'num_iters_per_epoch', 'iters']

            input_tokens = self.t5_tokenizer(
                samples["text_input"],#['where are the people standing?']
                # "a photo of ",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            # print(input_tokens.attention_mask.shape)# [4,10]
            output_tokens = self.t5_tokenizer(
                samples["answer"],#['balcony', 'porch', 'deck', 'platform', 'tower']
                #samples["text_output"],
                #samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)# 'input_ids','attention_mask' 全1
        ##################################
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []
            ##[1, 32, 2048]->[7, 32, 2048]
            for b, n in enumerate(samples["n_answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)#[7, 32, 2048]
            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            if self.use_adalink_T:
                inputs_embeds = inputs_embeds + self.adalink_T(inputs_embeds)#[7, 8, 2048]
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)# [7, 32+8, 2048] 

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,# torch.Size([12, 42, 2048])
                attention_mask=encoder_atts,# torch.Size([12, 42])
                decoder_attention_mask=output_tokens.attention_mask,# torch.Size([12, 4])
                return_dict=True,
                labels=targets,# torch.Size([12, 4])
            )

            loss = outputs.loss
            # wandb
            import wandb
            wandb.login(key="3d3950bf0197bb6a4f59246bd3ddeacd1ae2617d")     
            wandb.init(project="blip2_t5_okvqa", name=self.wandb_name) 
            wandb.log({"train_loss": loss})   
            
            return {"loss": loss}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        if self.use_adalink_I:
            inputs_t5 = inputs_t5 + self.adalink_I(inputs_t5)#[1, 32, 2048]->2048->[1, 32, 2048] 
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
        # with torch.cuda.amp.autocast():
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            if self.use_adalink_T:
                inputs_embeds = inputs_embeds + self.adalink_T(inputs_embeds)#[7, 8, 2048]
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        if self.use_adalink_I:
            inputs_t5 = inputs_t5 + self.adalink_I(inputs_t5)#[1, 32, 2048]->2048->[1, 32, 2048] 
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
        # with torch.cuda.amp.autocast():
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            if self.use_adalink_T:
                inputs_embeds = inputs_embeds + self.adalink_T(inputs_embeds)#[7, 8, 2048]
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_linear = cfg.get("freeze_linear", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        rank = cfg.get("ada_rank", 16)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        use_adalink_I=cfg.get("use_adalink_I", True)
        use_adalink_T=cfg.get("use_adalink_T", True)
        use_adalink_qformer=cfg.get("use_adalink_qformer", False)
        
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_linear=freeze_linear,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            rank=rank,
            use_adalink_I=use_adalink_I,
            use_adalink_T=use_adalink_T,
            use_adalink_qformer=use_adalink_qformer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
