from transformers import Trainer
import torch

class CustomDPOTrainer(Trainer):
    def __init__(self, model, ref_model, beta=0.1, *args, **kwargs):
        self.disable_dropout_in_model(model)
        self.disable_dropout_in_model(ref_model)
        super().__init__(model=model, *args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.ref_model.eval()


    def disable_dropout_in_model(self, model: torch.nn.Module):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

    def get_batch_logps(self, model, input_ids, attention_mask, prompt_lens, device='cuda:0'):
        model = model.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, vocab)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = input_ids  # (B, L)
        # gather token logp
        tgt_logprobs = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

        batch_size, seq_len = labels.size()

        # 1. index (B, L): [0,1,2,...L-1]
        arange_ids = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, seq_len)
        # 2. prompt_lens (B,) -> (B, 1)
        prompt_lens = prompt_lens.unsqueeze(1)
        # 3. mask
        resp_mask = (arange_ids >= prompt_lens)
        # 4. attention_mask (B, L)
        valid_mask = resp_mask & attention_mask.bool()

        # 5. sum response logp
        per_token_logp = tgt_logprobs * valid_mask  # (B, L)
        per_sample_logps = per_token_logp.sum(dim=1)  # (B,)

        return per_sample_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs dict，
        - chosen_input_ids, chosen_attention_mask
        - rejected_input_ids, rejected_attention_mask

        """

        prompt_lens = inputs["prompt_lens"]
        device = model.device
        logp_chosen = self.get_batch_logps(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"],
                                           prompt_lens, device=device) #-model(**chosen).loss
        logp_rejected = self.get_batch_logps(model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"],
                                             prompt_lens, device=device)
        with torch.no_grad():
            ref_logp_chosen = self.get_batch_logps(self.ref_model, inputs["chosen_input_ids"],
                                                   inputs["chosen_attention_mask"], prompt_lens, device=device) #-self.ref_model(**chosen).loss
            ref_logp_rejected = self.get_batch_logps(self.ref_model, inputs["rejected_input_ids"],
                                                     inputs["rejected_attention_mask"], prompt_lens, device=device)  #-self.ref_model(**rejected).loss


        dpo_logits = self.beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected))
        loss = -torch.nn.functional.logsigmoid(dpo_logits).mean()
        if return_outputs:
            return loss, inputs
        return loss
