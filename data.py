from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import ipdb


class WebDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index: Any) -> Any:
        return self.data[index]
    
    def __len__(self):
        return self.data.__len__()


def data_collator(batch, max_len):
    batch_input_ids, batch_labels, batch_attention_mask = [], [], []
    for row in batch:
        input_ids, labels, token_type_ids = row
        seq_len = input_ids.__len__()
        assert seq_len <= max_len, ipdb.set_trace()
        #比较input_ids长度(seq_len)与max_len，若超过，则对其进行截断；若小于，则用0进行填充
        #attention_mask:标识实际有效的输入部分
        if seq_len < max_len:
            input_ids = input_ids + [0]*(max_len-seq_len)
            labels = labels + [0]*(max_len-seq_len)
            token_type_ids = token_type_ids + [0]*(max_len-seq_len)
            attention_mask = [1]*seq_len + [0]*(max_len-seq_len)
        else: 
            attention_mask = [1]*seq_len 
        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        #batch_token_type_ids.append(token_type_ids)
        batch_attention_mask.append(attention_mask)
        #batch_is_next.append(int(is_next))
    return torch.tensor(batch_input_ids),\
            torch.tensor(batch_attention_mask),\
                torch.tensor(batch_labels)