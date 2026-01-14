from .selector import SelectPolicy
from ..datasets.jailbreak_datasets import JailbreakDataset
from ..utils import model_utils
from ..models.huggingface_model import WhiteBoxModelBase
import torch
import logging

class HiddenLayerLossSelector(SelectPolicy):
    """
    This class implements a selection policy based on the loss calculated between hidden layers of aligned and unaligned models.
    It selects instances from a set of parents based on the minimum loss calculated between the hidden layer representations,
    which can help identify jailbreak prompts that cause the largest divergence between aligned and unaligned models.
    """
    def __init__(self, target_model:WhiteBoxModelBase, unaligned_model:WhiteBoxModelBase, 
                 batch_size=None, is_universal=False):
        """
        Initialize the selector with target and unaligned models, and optional configuration settings.

        :param ~WhiteBoxModelBase target_model: The aligned target model whose hidden layers will be compared.
        :param ~WhiteBoxModelBase unaligned_model: The unaligned model whose hidden layers will be compared.
        :param int|None batch_size: The size of each batch for loss calculation. If None, batch_size will be the same as the size of dataset (default None).
        :param bool is_universal: If True, considers the loss of all instances with the same jailbreak_prompt together (default False).
        """
        assert isinstance(target_model, WhiteBoxModelBase)
        assert isinstance(unaligned_model, WhiteBoxModelBase)
        self.target_model = target_model
        self.unaligned_model = unaligned_model
        self.batch_size = batch_size
        self.is_universal = is_universal

    def _get_hidden_layer_loss(self, target_input_ids, unaligned_input_ids, target_attention_mask=None, unaligned_attention_mask=None):
        """
        Calculate loss based on the difference between hidden layer representations of aligned and unaligned models.
        Uses all hidden layers from both models to compute a comprehensive loss.
        
        :param torch.Tensor target_input_ids: The input token IDs for the target (aligned) model.
        :param torch.Tensor unaligned_input_ids: The input token IDs for the unaligned model.
        :param torch.Tensor|None target_attention_mask: Attention mask for padded sequences in target model.
        :param torch.Tensor|None unaligned_attention_mask: Attention mask for padded sequences in unaligned model.
        :return torch.Tensor: The calculated loss values.
        """
        # Forward pass through aligned model (target_model) with output_hidden_states=True to get hidden states
        aligned_outputs = self.target_model.model(
            input_ids=target_input_ids, 
            attention_mask=target_attention_mask if target_attention_mask is not None else None,
            output_hidden_states=True
        )
        
        # Forward pass through unaligned model with the appropriate input
        unaligned_outputs = self.unaligned_model.model(
            input_ids=unaligned_input_ids,
            attention_mask=unaligned_attention_mask if unaligned_attention_mask is not None else None,
            output_hidden_states=True
        )
        
        # Get all hidden states from both models
        aligned_hidden_states_list = aligned_outputs.hidden_states
        unaligned_hidden_states_list = unaligned_outputs.hidden_states
        
        # Get the last hidden state from both models
        # Get the last hidden state from the last layer, and select only the last token
        aligned_hidden_states = aligned_hidden_states_list[-1][:, -1, :]
        unaligned_hidden_states = unaligned_hidden_states_list[-1][:, -1, :]
        
        # Calculate Cosine Similarity along the hidden dimension
        cosine_sim = torch.nn.functional.cosine_similarity(
            aligned_hidden_states, 
            unaligned_hidden_states, 
        )

        # Calculate the loss as 1 - similarity
        total_loss = 1 - cosine_sim
        
        return total_loss

    def select(self, dataset)->JailbreakDataset:
        """
        Selects instances from the dataset based on the calculated hidden layer loss between aligned and unaligned models.

        :param ~JailbreakDataset dataset: The dataset from which instances are to be selected.
        :return ~JailbreakDataset: A new dataset containing selected instances with minimum hidden layer loss.
        """
        if not self.is_universal and len(dataset.group_by_parents()) > 1:
            # Treat the case where is_universal=False as a special case of the True case
            return JailbreakDataset.merge([self.select(JailbreakDataset(group)) for group in dataset.group_by_parents()])

        if self.batch_size is None:
            batches = [dataset]
        else:
            batches = [dataset[i: i+self.batch_size] for i in range(0, len(dataset), self.batch_size)]

        # Calculate loss based on hidden layer representations
        with torch.no_grad():
            for batch in batches:
                B = len(batch)
                
                # Encode jailbreak queries for both target and unaligned models
                batch_target_input_ids = []
                batch_unaligned_input_ids = []
                
                for instance in batch:
                    # Encode jailbreak query for target model
                    target_input_ids, _, _, _ = model_utils.encode_trace(
                        self.target_model, 
                        instance.query, 
                        instance.jailbreak_prompt, 
                        ""  # Empty response to get prompt encoding only
                    )
                    batch_target_input_ids.append(target_input_ids)
                    
                    # Encode jailbreak query for unaligned model
                    unaligned_input_ids, _, _, _ = model_utils.encode_trace(
                        self.unaligned_model, 
                        instance.query, 
                        "{query}", 
                        ""  # Empty response to get prompt encoding only
                    )
                    batch_unaligned_input_ids.append(unaligned_input_ids)
                
                # Pad and stack for both models
                batch_target_input_ids = model_utils.pad_and_stack(batch_target_input_ids, self.target_model.pad_token_id)
                batch_unaligned_input_ids = model_utils.pad_and_stack(batch_unaligned_input_ids, self.unaligned_model.pad_token_id)
                
                # Compute attention masks
                target_attention_mask = (batch_target_input_ids != self.target_model.pad_token_id).float()
                unaligned_attention_mask = (batch_unaligned_input_ids != self.unaligned_model.pad_token_id).float()
                
                # Compute hidden layer loss between aligned and unaligned models
                batch_loss = self._get_hidden_layer_loss(
                    batch_target_input_ids,
                    batch_unaligned_input_ids,
                    target_attention_mask,
                    unaligned_attention_mask
                )
                
                # Store loss value for each instance
                for idx, instance in enumerate(batch):
                    instance._loss = batch_loss[idx].item()
            
        # select
        best_group = None
        best_loss = None
        for group in dataset.group_by(lambda x: x.jailbreak_prompt):
            total_loss = sum([instance._loss for instance in group])
            if best_loss is None or total_loss < best_loss:
                best_loss = total_loss
                best_group = group
        logging.info(f'Loss selection: best loss = {best_loss}')
        logging.info(f'Loss Selection: best jailbreak prompt = `{best_group[0].jailbreak_prompt}`')

        return JailbreakDataset(best_group)