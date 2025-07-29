from transformers import Trainer
import torch

class CustomDPOTrainer(Trainer):
    def __init__(self, ref_model, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.ref_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        inputs dict，
        - chosen_input_ids, chosen_attention_mask
        - rejected_input_ids, rejected_attention_mask

        """
        chosen = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "labels": inputs["chosen_input_ids"]
        }
        rejected = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "labels": inputs["rejected_input_ids"]
        }

        # 当前模型概率
        logp_chosen = -model(**chosen).loss
        logp_rejected = -model(**rejected).loss
        # 参考模型概率
        with torch.no_grad():
            ref_logp_chosen = -self.ref_model(**chosen).loss
            ref_logp_rejected = -self.ref_model(**rejected).loss

        dpo_logits = self.beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected))
        loss = -torch.nn.functional.logsigmoid(dpo_logits)
        if return_outputs:
            return loss, inputs
        return loss