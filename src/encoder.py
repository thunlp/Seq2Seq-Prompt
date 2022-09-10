"""Prompt tuning encoder."""

import torch


class PromptEncoder(object):
    def __init__(self):
        self.vocab_size = 32100
        self.start_idx = self.vocab_size - 100
        self.pattern_convert = []
        self.init_success = False

    def add_prompt_tokens(self, origin_ids):
        # Convert tokens in manual prompt / label to unused tokens
        # Note that `AlbertTokenizer` or `RobertaTokenizer` doesn't have a `vocab` attribute
        for origin_id in origin_ids:
            self.pattern_convert.append((origin_id, self.start_idx))
            self.start_idx += 1

    def get_prompt_tokens(self, begin, len):
        return [self.pattern_convert[i][1] for i in range(begin, begin + len)]

    def init_embed(self, model):
        w = model.get_input_embeddings().weight.data
        for origin_id, convert_id in self.pattern_convert:
            w[convert_id] = w[origin_id]

    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = [convert_id for _, convert_id in self.pattern_convert]
        grad_mask = torch.zeros((32128, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def add_reverse_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = [convert_id for _, convert_id in self.pattern_convert]
        grad_mask = torch.ones((32128, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 0.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)
