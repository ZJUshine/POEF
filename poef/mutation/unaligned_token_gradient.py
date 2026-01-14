from .mutation_base import MutationBase
from ..utils import model_utils
from ..models.huggingface_model import WhiteBoxModelBase
from ..datasets.instance import Instance
from ..datasets.jailbreak_datasets import JailbreakDataset
import random
import torch
import torch.nn.functional as F
from typing import Optional


class MutationUnalignedTokenGradient(MutationBase):
    def __init__(
            self, 
            attack_model: WhiteBoxModelBase,
            unaligned_model: WhiteBoxModelBase,
            num_turb_sample: Optional[int] = 512,
            top_k: Optional[int] = 256,
            avoid_unreadable_chars: Optional[bool] = True,
            is_universal: Optional[bool] = False,
            hidden_layer_idx: Optional[int] = -1):
        """
        Initializes the MutationUnalignedTokenGradient.

        :param WhiteBoxModelBase attack_model: Model used for tokenization and processing.
        :param WhiteBoxModelBase unaligned_model: Unaligned model used for gradient computation.
        :param int num_turb_sample: Number of mutant samples generated per instance. Defaults to 512.
        :param int top_k: Randomly select the target mutant token from the top_k with the smallest gradient values at each position.
            Defaults to 256.
        :param bool avoid_unreadable_chars: Whether to avoid generating unreadable characters. Defaults to True.
        :param bool is_universal: Whether a shared jailbreak prompt is optimized for all instances. Defaults to False.
        :param int hidden_layer_idx: Index of the hidden layer to use for similarity calculation. Defaults to -1 (last layer).
        """
        self.attack_model = attack_model
        self.unaligned_model = unaligned_model
        self.num_turb_sample = num_turb_sample
        self.top_k = top_k
        self.avoid_unreadable_chars = avoid_unreadable_chars
        self.is_universal = is_universal
        self.hidden_layer_idx = hidden_layer_idx
    
    def calculate_hidden_similarity_gradient(self, attack_model, unaligned_model, input_ids, unaligned_input_ids, target_slice):
        """
        Calculate token gradients using the similarity between hidden layers of attack_model and unaligned_model.
        
        :param WhiteBoxModelBase attack_model: The attack model to get hidden states from
        :param WhiteBoxModelBase unaligned_model: The unaligned model to get hidden states from
        :param torch.Tensor input_ids: Input token IDs tensor
        :param slice target_slice: Slice indicating response region
        :return torch.Tensor: Gradient tensor
        """
        assert input_ids.size(0) == 1
        L = input_ids.size(1)   # input_ids: 1 * L
        V = attack_model.vocab_size

        # Convert input_ids to one-hot representation with gradients
        one_hot_input = F.one_hot(input_ids, num_classes=V).to(attack_model.dtype)    # 1 * L * V
        one_hot_input.requires_grad = True
        unaligned_one_hot_input = F.one_hot(unaligned_input_ids, num_classes=V).to(unaligned_model.dtype)    # 1 * L * V
        unaligned_one_hot_input.requires_grad = True
        
        # Get embeddings
        attack_embed_matrix = attack_model.embed_layer.weight   # V * D
        unaligned_embed_matrix = unaligned_model.embed_layer.weight   # V * D
        
        attack_inputs_embeds = torch.matmul(one_hot_input, attack_embed_matrix)   # 1 * L * D
        unaligned_inputs_embeds = torch.matmul(unaligned_one_hot_input, unaligned_embed_matrix)   # 1 * L * D
        
        # Get hidden states from both models - without torch.no_grad() to allow gradient flow
        attack_outputs = attack_model(inputs_embeds=attack_inputs_embeds, output_hidden_states=True)
        attack_hidden_states = attack_outputs.hidden_states[self.hidden_layer_idx]  # 1 * L * D

        unaligned_outputs = unaligned_model(inputs_embeds=unaligned_inputs_embeds, output_hidden_states=True)
        unaligned_hidden_states = unaligned_outputs.hidden_states[self.hidden_layer_idx]  # 1 * L * D
        
        # Calculate cosine similarity between hidden states
        # We want to maximize this similarity, so we'll use it as our loss
        similarity = F.cosine_similarity(attack_hidden_states[:, -1, :], unaligned_hidden_states[:, -1, :])  # 1 * L
        loss = -similarity
        # Backward pass to get gradients
        loss.backward()
        
        return one_hot_input.grad    # 1 * L * V
        
    def __call__(
            self,
            jailbreak_dataset: JailbreakDataset
            )->JailbreakDataset:
        """
        Mutates the jailbreak_prompt in the sample based on the unaligned model's token gradient.
        
        :param JailbreakDataset jailbreak_dataset: Dataset for the attack.
        :return: A mutated dataset with the jailbreak prompt based on the unaligned model's token gradient.
        :rtype: JailbreakDataset
        
        .. note::
            - num_turb_sample: Number of mutant samples generated per instance.
            - top_k: Each mutation target is selected from the top_k tokens with the smallest gradient values at each position.
            - is_universal: Whether the jailbreak prompt is shared across all samples. If true, it uses the first sample in the jailbreak_dataset.
        """
        
        # Handle is_universal=False as a special case (multiple datasets of size 1)
        if not self.is_universal and len(jailbreak_dataset) > 1:
            ans = []
            for instance in jailbreak_dataset:
                new_samples = self(JailbreakDataset([instance]))
                ans.append(new_samples)
            return JailbreakDataset.merge(ans)
        # The rest of the implementation assumes is_universal=True

        # Tokenize
        universal_prompt_ids = None
        for instance in jailbreak_dataset:
            if isinstance(instance.reference_responses, str):
                ref_resp = instance.reference_responses
            else:
                ref_resp = instance.reference_responses[0]
            # Build the complete string and annotate where prompt and target positions are
            instance.jailbreak_prompt = jailbreak_dataset[0].jailbreak_prompt   # Use the first sample as reference
            input_ids, query_slice, jbp_slices, response_slice = model_utils.encode_trace(self.attack_model, instance.query, instance.jailbreak_prompt, ref_resp)
            instance._input_ids = input_ids
            instance._query_slice = query_slice
            instance._jbp_slices = jbp_slices
            instance._response_slice = response_slice
            unaligned_input_ids, unaligned_query_slice, unaligned_jbp_slices, unaligned_response_slice = model_utils.encode_trace(self.unaligned_model, instance.query, '{query}', ref_resp)
            instance._unaligned_input_ids = unaligned_input_ids
            instance._unaligned_query_slice = unaligned_query_slice
            instance._unaligned_jbp_slices = unaligned_jbp_slices
            instance._unaligned_response_slice = unaligned_response_slice

            if universal_prompt_ids is not None:
                instance._input_ids[:, jbp_slices[0]] = universal_prompt_ids[0]
                instance._input_ids[:, jbp_slices[1]] = universal_prompt_ids[1]
            else:
                universal_prompt_ids = [input_ids[:, jbp_slices[0]], input_ids[:, jbp_slices[1]]]

        # Calculate token gradient using unaligned model
        # For each sample, calculate token gradient for the jailbreak_prompt part, normalize, and sum
        jbp_token_grad = 0   # 1 * L1 * V
        for instance in jailbreak_dataset:
            # Use hidden layer similarity as gradient source
            token_grad = self.calculate_hidden_similarity_gradient(
                self.attack_model, 
                self.unaligned_model, 
                instance._input_ids,
                instance._unaligned_input_ids,
                instance._response_slice
            )   # 1 * L * V
                
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)   # 1 * L * V
            jbp_token_grad += torch.cat([token_grad[:, instance._jbp_slices[0]], token_grad[:, instance._jbp_slices[1]]], dim=1)    # 1 * L1 * V
        L1 = jbp_token_grad.size(1)
        V = jbp_token_grad.size(2)
          
        # Generate variants
        score_tensor = -jbp_token_grad
        if self.avoid_unreadable_chars:
            ignored_ids = model_utils.get_non_english_word_ids(self.attack_model)
            for token_ids in ignored_ids:
                score_tensor[:,:,token_ids] = float('-inf')
        top_k_indices = torch.topk(score_tensor, dim=2, k=self.top_k).indices
        with torch.no_grad():
            # Generate perturbed prompts
            turbed_prompt_ids_list = []
            for _ in range(self.num_turb_sample):
                new_prompt_ids = [universal_prompt_ids[0].clone(), universal_prompt_ids[1].clone()]   # [1 * L11, 1 * L12]; L11+L12==L1
                # Handle special case where first token of jailbreak_prompt is bos
                if universal_prompt_ids[0].size(1) >= 1 and universal_prompt_ids[0][0, 0] == self.attack_model.bos_token_id:
                    rel_idx = random.randint(1, L1-1)
                else:
                    rel_idx = random.randint(0, L1-1)   # Which token in jailbreak_prompt to replace
                # Randomly select a new token_id
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, self.top_k-1)]
                # Replace the token
                if rel_idx < new_prompt_ids[0].size(1):
                    new_prompt_ids[0][0][rel_idx] = new_token_id
                else:
                    new_prompt_ids[1][0][rel_idx-new_prompt_ids[0].size(1)] = new_token_id
                turbed_prompt_ids_list.append(new_prompt_ids)
                
            # Generate new dataset with perturbed prompts
            new_dataset = []
            for instance in jailbreak_dataset:
                new_instance_list = []
                for new_prompt_ids in turbed_prompt_ids_list:
                    new_input_ids = instance._input_ids.clone()
                    new_input_ids[:, instance._jbp_slices[0]] = new_prompt_ids[0]
                    new_input_ids[:, instance._jbp_slices[1]] = new_prompt_ids[1]
                    _, _, jailbreak_prompt, _ = model_utils.decode_trace(self.attack_model, new_input_ids, instance._query_slice, instance._jbp_slices, instance._response_slice)
                    if '\r' in jailbreak_prompt:
                        breakpoint()
                    new_instance = Instance(
                        query=instance.query,
                        jailbreak_prompt=jailbreak_prompt,
                        reference_responses=instance.reference_responses,
                        parents=[instance]
                    )
                    new_instance_list.append(new_instance)
                instance.children = new_instance_list
                new_dataset.extend(new_instance_list)
            
            # Clean up tensors to free GPU memory
            for instance in jailbreak_dataset:
                instance.delete('_input_ids')
            
            return JailbreakDataset(new_dataset) 