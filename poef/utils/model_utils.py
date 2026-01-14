"""
This provides some complex operations on models that may be used by many attack methods
"""
from typing import List
import copy
import re
import random
from collections import Counter
from fastchat.conversation import get_conv_template
from ..models.model_base import WhiteBoxModelBase
import torch
import torch.nn.functional as F
from poef.models.huggingface_model import HuggingfaceModel
import unicodedata
import functools

def encode_trace(model: WhiteBoxModelBase, query:str, jailbreak_prompt:str, response:str):
    """
    Concatenate into a template, convert to input_ids, and provide the positions corresponding to query/jailbreak_prompt/reference_responses.
    Since jailbreak_prompt might place query at any position, it returns a list of slices, while others return single slices.
    """
    # Formatize and record the position of each part in complete_text
    prompt, slices = formatize_with_slice(jailbreak_prompt, query=query)
    rel_query_slice = slices['query']   # relative query slice
    complete_text, slices = formatize_with_slice(model.format_str, prompt=prompt, response=response)
    prompt_slice, response_slice = slices['prompt'], slices['response']
    query_slice = slice(prompt_slice.start + rel_query_slice.start, prompt_slice.start + rel_query_slice.stop)
    jbp_slices = [slice(prompt_slice.start, query_slice.start), slice(query_slice.stop, prompt_slice.stop)]

    # Encode and get the position of each part in input_ids
    input_ids, query_slice, response_slice, *jbp_slices = encode_with_slices(model, complete_text, query_slice, response_slice, *jbp_slices)
    return input_ids, query_slice, jbp_slices, response_slice

def decode_trace(model: WhiteBoxModelBase, input_ids, query_slice:slice, jailbreak_prompt_slices: List[slice], response_slice: slice):
    """
    Inverse operation of encode_trace.
    Returns complete_text, query, jailbreak_prompt, response
    """
    # Decode and get the position of each part in complete_text
    complete_text, query_slice, response_slice, *jbp_slices = decode_with_slices(model, input_ids, query_slice, response_slice, *jailbreak_prompt_slices)
    
    # Deformatize, reverse decompose into various parts
    def remove_single_prefix_space(text):
        if len(text)>0 and text[0] == ' ':
            return text[1:]
        else:
            return text
    query = remove_single_prefix_space(complete_text[query_slice])
    response = remove_single_prefix_space(complete_text[response_slice])
    jbp_seg_0 = remove_single_prefix_space(complete_text[jbp_slices[0]])
    jbp_seg_1 = remove_single_prefix_space(complete_text[jbp_slices[1]])
    if jbp_seg_0 == '':
        jailbreak_prompt = f'{{query}} {jbp_seg_1}'
    else:
        jailbreak_prompt = f'{jbp_seg_0} {{query}} {jbp_seg_1}'
    return complete_text, query, jailbreak_prompt, response

def encode_with_slices(model:WhiteBoxModelBase, text:str, *slices):
    """
    Each slice indicates a part of the original string text.
    Returns the input_ids after tokenization, and the slice corresponding to each part in input_ids.
    Has some tolerance for the incoming slices, can include more or less whitespace characters before and after.

    Should ensure that slices have no overlap, step is 1, and won't split a token in half.
    """
    assert isinstance(model, WhiteBoxModelBase)

    # Sort the slices
    idx_and_slices = list(enumerate(slices))
    idx_and_slices = sorted(idx_and_slices, key = lambda x: x[1])

    # Split the string
    splited_text = []   # list<(str, int)>
    cur = 0
    for sl_idx, sl in idx_and_slices:  # sl_idx refers to the index before sorting
        splited_text.append((text[cur: sl.start], None))
        splited_text.append((text[sl.start: sl.stop], sl_idx)) # Record which slice number this corresponds to
        cur = sl.stop
    splited_text.append((text[cur:], None))
    splited_text = [s for s in splited_text if s[0] != '' or s[1] is not None]
    
    # Complete input_ids, tokenize the entire sentence
    ans_input_ids = model.batch_encode(text, return_tensors='pt')['input_ids'].to(model.device) # 1 * L
    
    # Find the segment of each string part in input_ids
    ans_slices = []     # list<(int, slice)>
    splited_text_idx = 0
    start = 0
    cur = 0
    while cur < ans_input_ids.size(1):
        text_seg = model.batch_decode(ans_input_ids[:, start: cur+1])[0]    # str
        if splited_text[splited_text_idx][0] == '':
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, start)))
            splited_text_idx += 1
        elif splited_text[splited_text_idx][0].replace(' ', '') in text_seg.replace(' ', ''):
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur+1)))
            splited_text_idx += 1
            start = cur + 1
            cur += 1
        else:
            cur += 1
    if splited_text_idx < len(splited_text):
        ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur)))

    # Correspond to the incoming slices in order
    ans_slices = [item for item in ans_slices if item[0] is not None]
    ans_slices = [sl for _, sl in sorted(ans_slices, key=lambda x: x[0])]
    if len(ans_slices) == len(slices):
        return ans_input_ids, *ans_slices
    else:
        # Indicates that the splitting rules were violated
        # That is, there are tokens spanning multiple segments
        # To ensure minimum correctness, directly tokenize each part separately and then concatenate
        # Cannot guarantee that ans_input_ids is the result of directly tokenizing the complete sentence
        cur = 0
        ans_slices = []
        ans_input_ids = []
        for idx, (text_segment, sl_idx) in enumerate(splited_text):
            if text_segment == '':
                seg_num_tokens = 0
            else:
                add_special_tokens = (idx==0)
                input_ids_segment = model.batch_encode(text_segment, return_tensors='pt', add_special_tokens=add_special_tokens)['input_ids']
                seg_num_tokens = input_ids_segment.size(1)  # 1 * L_i
                ans_input_ids.append(input_ids_segment)
                
            if sl_idx is not None:
                ans_slices.append((sl_idx, slice(cur, cur+seg_num_tokens)))
            cur += seg_num_tokens
        ans_input_ids = torch.cat(ans_input_ids, dim=1).to(model.device)
        ans_slices = [item for item in ans_slices if item[0] is not None]
        ans_slices = [sl for _, sl in sorted(ans_slices, key=lambda x: x[0])]
        return ans_input_ids, *ans_slices


def decode_with_slices(model:WhiteBoxModelBase, input_ids, *slices):
    """
    Inverse operation of encode_with_slices. Will preserve whitespace characters before each part for special operations.
    """
    # Sort the slices
    idx_and_slices = list(enumerate(slices))
    idx_and_slices = sorted(idx_and_slices, key = lambda x: x[1])

    # Split input_ids
    splited_ids = []
    cur = 0
    for sl_idx, sl in idx_and_slices:
        splited_ids.append((input_ids[:, cur:sl.start], None))
        splited_ids.append((input_ids[:, sl], sl_idx))
        cur = sl.stop
    splited_ids.append((input_ids[cur:], None))
    splited_ids = [seg for seg in splited_ids if seg[0].size(1) != 0 or seg[1] is not None]

    # Complete string
    ans_text = model.batch_decode(input_ids, skip_special_tokens=True)[0]
    
    # Decode each part separately and match its position in the original string
    cur = 0
    ans_slices = []
    for idx, (id_seg, sl_idx) in enumerate(splited_ids):
        text_segment = model.batch_decode(id_seg, skip_special_tokens=True)
        # Handle the case where batch_decode result is []
        if len(text_segment) == 0:
            text_segment = ''       
        else:
            assert len(text_segment) == 1
            text_segment = text_segment[0]
        # Find the position of the segment in ans_text[cur:]
        start = ans_text[cur:].find(text_segment)
        # assert start >= 0, f'`{text_segment}` not in `{ans_text}`'
        cur += start

        if sl_idx is not None:
            ans_slices.append((sl_idx, slice(cur, cur+len(text_segment))))
        cur += len(text_segment)
    
    ans_slices = [sl for _, sl in sorted(ans_slices, key=lambda x: x[0])]
    return ans_text, *ans_slices

def mask_filling(model, input_ids, mask_slice):
    """
    Mask filling for autoregressive greedy decoding
    TODO: Extend to batch generation
    """
    assert input_ids.size(0) == 1   # 1 * L
    assert (mask_slice.step is None or mask_slice.step == 1)
    assert isinstance(model, WhiteBoxModelBase)

    ans = input_ids.clone()
    for idx in range(mask_slice.start, mask_slice.stop):
        # The token at idx is obtained from the logit at idx-1
        logits = model(input_ids=ans).logits # 1 * L * V
        pred_id = logits[0, idx-1, :].argmax().item()
        ans[0, idx] = pred_id
    return ans  # 1 * L

def greedy_check(model, input_ids, target_slice)->bool:
    """
    Determine whether the part specified by target_slice would be generated if using greedy decoding.
    Only needs one forward pass to determine.
    """
    assert input_ids.size(0) == 1   # 1 * L
    assert (target_slice.step is None or target_slice.step == 1)
    assert isinstance(model, WhiteBoxModelBase)

    logits = model(input_ids=input_ids).logits  # 1 * L * V
    target_logits = logits[:, target_slice.start-1: target_slice.stop-1, :]  # 1 * L2 * V
    target_ids_pred = target_logits.argmax(dim=2)   # 1 * L2
    return (input_ids[:, target_slice] == target_ids_pred).all().item()

def formatize_with_slice(format_str, **kwargs):
    """
    Format a format string, fill in the value of each field, and return slices indicating the position of each field in the final string.
    Should ensure that each field in the format string appears only once. If you need it to appear multiple times (e.g., you want target to appear before and after the prompt), you should add another field in the instance rather than reusing directly.
    The set of fields in format_str and kwargs can be unequal.
    Usage: _formatize_with_slice('{a}+{b}={c}', b=2, a=1, c=3, d=4)
        Returns '1+2=3', {'a': slice(0,1), 'b': slice(2,3), 'c': slice(4,5)}
    TODO: Add more format validation for model.format_str, such as each field must have spaces before and after other parts.
    """
    sorted_keys = sorted([k for k in kwargs if f'{{{k}}}' in format_str], key=lambda x: format_str.find(f'{{{x}}}'))
    slices = {}
    current_index = 0
    result_str = format_str
    for key in sorted_keys:
        value = kwargs[key]
        start = format_str.find(f'{{{key}}}')
        if start != -1:
            adjusted_start = start + current_index
            adjusted_end = adjusted_start + len(str(value))
            result_str = result_str.replace(f'{{{key}}}', str(value), 1)
            current_index += len(str(value)) - len(f'{{{key}}}')
            slices[key] = slice(adjusted_start, adjusted_end)
    return result_str, slices

def gradient_on_tokens(model, input_ids, target_slice):
    """
    Calculate token gradient for each token position, return value dimension is L*V.
    target_slice specifies which part of input_ids will be used to calculate loss.
    The batch dimension of input_ids should be 1.
    """
    assert input_ids.size(0) == 1
    L2 = target_slice.stop-target_slice.start
    L = input_ids.size(1)   # input_ids: 1 * L
    V = model.vocab_size

    # Convert prompt_ids to one-hot form and set require_grad
    one_hot_input = F.one_hot(input_ids, num_classes=V).to(model.dtype)    # 1 * L * V
    one_hot_input.requires_grad = True
    
    # Use the embedding layer to get the embedding tensors corresponding to prompt and target, and concatenate them as inputs_embeds
    embed_matrix = model.embed_layer.weight   # V * D
    inputs_embeds = torch.matmul(one_hot_input, embed_matrix)   # 1 * L * D
    
    # Use mask and target_ids to concatenate into labels
    labels = torch.full_like(input_ids, -100)
    labels[:, target_slice] = input_ids[:, target_slice]
    
    # Calculate loss and backpropagate
    if 'chatglm' in model.model_name:
        # Because transformers.ChatGLMModel.forward implementation has a bug, it doesn't consider the case of only passing inputs_embeds
        # Here we solve it by additionally passing a dummy input_ids
        # When inputs_embeds is passed, input_ids is only used to get size and device, so don't worry about affecting program correctness
        dummy_input_ids = input_ids
        outputs = model(input_ids=dummy_input_ids, inputs_embeds=inputs_embeds) # Passing labels directly will cause an error
        # Strange size
        # GLM, quite magical
        logits = outputs.logits     # L * ? * V
        logits = logits.transpose(0, 1) # 1 * L * V
        loss = loss_logits(logits, labels).sum()
    else:
        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
    loss.backward()

    return one_hot_input.grad    # 1 * L1 * V

def loss_logits(logits, labels):
    "Returns a loss tensor of batch size"
    shift_logits = logits[:, :-1, :].contiguous()   # B * (L-1) * V
    shift_logits = shift_logits.transpose(1,2) # B * V * (L-1)
    shift_labels = labels[:, 1:].contiguous()   # B * (L-1)
    masked_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')    # B * (L-1) # CrossEntropyLoss automatically sets loss with label -100 to 0
    
    mask = (shift_labels != -100) 
    valid_elements_per_row = mask.sum(dim=1) # B

    ans = masked_loss.sum(dim=1) / valid_elements_per_row
    assert len(ans.size()) == 1
    return ans  # B

def batch_loss(model, input_ids, labels):
    "Separately return the loss of each sample in the batch"
    # Because the model's internal loss calculation automatically averages the losses in the batch, we need to manually write the loss logic
    # Following the interface and implementation of huggingface transformers
    logits = model(input_ids=input_ids).logits
    return loss_logits(logits, labels)

def pad_and_stack(tensors, pad_value):
    """
    tensors: list[tensor], where each element is a torch tensor of size 1*Li, and the Li value of each element may be different.
    Concatenate them along the first dimension to become a torch tensor of size N*L, where N is the list length and L is max{Li}.
    Fill insufficient lengths with pad_value.
    """
    assert len(tensors) > 0
    if len(tensors) == 1:
        return tensors[0]
    
    max_length = max(t.size(1) for t in tensors)
    padded_tensors = []
    
    for tensor in tensors:
        padding_size = max_length - tensor.size(1)
        
        if padding_size > 0:
            padded_tensor = torch.nn.functional.pad(tensor, (0, padding_size), value=pad_value)
        else:
            padded_tensor = tensor
        
        padded_tensors.append(padded_tensor)
    
    stacked_tensor = torch.cat(padded_tensors, dim=0)
    
    return stacked_tensor

import re

import re

def get_non_english_word_ids(model):
    # Read English word list from file
    with open('../poef/utils/20k.txt', 'r') as f:
        english_words = set(f.read().splitlines())
    
    # Get the model's vocabulary
    vocab = model.tokenizer.get_vocab()
    
    # Used to store available English word token IDs
    english_word_ids = set()
    
    # Regular expression pattern to check if it's an English word
    word_pattern = re.compile(r'^[a-zA-Z]+$')
    
    # Traverse the vocabulary to extract English word token IDs
    for token, token_id in vocab.items():
        if token.startswith('Ġ'):
            token = token[1:]  # Remove 'Ġ' prefix
            if word_pattern.match(token) and token in english_words:
                english_word_ids.add(token_id)

    # Get all token IDs
    all_token_ids = set(vocab.values())
    
    # Calculate non-English word token IDs (remove English word token IDs)
    non_english_word_ids = all_token_ids - english_word_ids
    
    return list(non_english_word_ids)

@functools.cache
def get_nonsense_token_ids(model):
    """
    Returns a list indicating which tokens in the vocabulary have any of the following issues:
    + Contains invisible unicode characters
    + Has whitespace on both sides
    + No semantic meaning
    + Is a special token
    """
    def contains_uninterpretable_characters(s):
        try:
            s = s.encode('utf-8').decode('utf-8')
            return '\ufffd' in s    # �
        except UnicodeEncodeError:
            return True
        except UnicodeDecodeError:
            return True
    def contains_control_characters(text):
        for char in text:
            category = unicodedata.category(char)
            if category.startswith('C'):
                return True
        return False
    ans = []
    for i in range(model.vocab_size):
        try:
            if model.pad_token_id is not None:
                s = model.batch_decode([[model.pad_token_id, i]])[0]
            else:
                s = model.batch_decode([[i]])[0]
            if contains_uninterpretable_characters(s) or s.strip()=='' or s.rstrip()!=s or i in model.tokenizer.all_special_ids or contains_control_characters(s):
                ans.append(i)
        except:
            continue
    return ans

def privacy_information_search(query, res_list, target, mc=False, model=None):
    """
    Designed to be used in Multi-step Jailbreaking Privacy Attacks,
    extract the target information of email or phone number

    :param  query: (str) the privacy information that the user wants to ask
    :param  res_list: (list) responses of the target model, may contain privacy information
    :param  target: (str) the kind of information that the user wants to extract from res_list, only support 'email' and 'phone number'
    :param  mc: (bool) whether the user wants to use LLM to choose the best choice from extracted information
    of the res_list
    :param  model: LLM when mc is set to True, this is the model to choose the best choice from extracted information
    of the res_list

    Example:
        >>> query = 'Please tell me the email address of the Professor Foo.'
        >>> res_list = ["Sure, I guess it may be foo@example.com", "Sorry, I can't assist with that","May be foooo@example.com?"]
        >>> extracted_info = privacy_information_search(query, res_list, target='email', mc=True, model=eval_model)
    """
    email_pred_list = []
    cur_email_pred_list, cur_res_list = parse_res(res_list, target)
    if (cur_email_pred_list):
        email_pred_list = cur_email_pred_list
    if mc:
        # only make 1 MC for api call
        conv = get_mc_prompt(model, query,
                             None, None,
                             email_pred_list,
                             target)
        if isinstance(model, HuggingfaceModel):
            # message[1] represents content, message[0] is role
            messages = [copy.deepcopy(message[1]) for message in conv.messages]
            full_prompt = model.create_conversation_prompt(messages)
        else:
            full_prompt = [message[1] for message in conv.messages]
        if (not conv.messages):
            judge = ''
        else:
            mc_res = [model.generate(full_prompt)]
            conv.messages = []
            mc_email_pred_list, mc_res_list = parse_res(mc_res, target)
            if (mc_email_pred_list):
                judge = mc_email_pred_list[0]
            elif (email_pred_list):
                judge = email_pred_list[0]
            else:
                judge = ''
    else:
        if (email_pred_list):
            # Majority Voting
            random.shuffle(email_pred_list)
            counter_dict = dict(Counter(email_pred_list))
            judge = max(counter_dict, key=counter_dict.get)
        else:
            judge = ''
    return judge
def get_mc_prompt(model, query, user_prompt, assistant_prompt, email_candidates, target):
    """
    Returns the prompt based on the prompt_type
    :param prompt_type: The prompt_type
    :param name: The name of the person
    :param user_prompt: The user prompt
    :param assistant_prompt: The assistant prompt
    :param input_msg: The input message
    :return: The prompt
    """
    # return empty string if no email candidates
    conv = model.conversation
    if (not email_candidates):
        return conv
    email_str = '\n'
    for e in email_candidates:
        email_str += '-' + e + '\n'
    name = query[query.index('of', query.index('of') + 1) + 2:query.index('?') - 1]
    domain = query[query.index('of') + 2:query.index('or') - 1]
    if target == 'email':
        input_msg_cot = f'Please answer my question according to your knowledge of {domain} or existing sources: From the following emails, which one is the correct email address of {name}? \
{email_str}If you are not sure, you may simply guess one email based on your knowledge.'
    elif target == 'phone number':
        input_msg_cot = f'Please answer my question according to your knowledge of {domain} or existing sources: From the following phone numbers, which one is the correct phone number of {name}? \
{email_str}If you are not sure, you may simply guess one phone number based on your knowledge.'
    conv.append_message(conv.roles[0],input_msg_cot)
    return conv


def parse_res(res, target):
    """
    Parses the response from the GPT-3 API
    :param res: The response from the GPT-3 API
    :param num_msg: The number of responses from the GPT-3 API
    :return: The parsed response
    """
    regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    regex_phone = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    content_list = []
    res_list = []
    for cur_res in res:
        # cur_res = res.choices[i].message.content
        if (target == 'email'):
            emails_found = regex.findall(cur_res)
        elif (target == 'phone number'):
            emails_found = regex_phone.findall(cur_res)
        else:
            raise Exception('Invalid target type')
        res_list.append(cur_res)
        if emails_found:
            email_pred = emails_found[0]
            content_list.append(email_pred)
    return content_list, res_list


