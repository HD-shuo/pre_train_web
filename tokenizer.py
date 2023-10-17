import torch 
import pandas as pd
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model 
from pandas import read_parquet
from torch.utils.data import dataloader,dataset
import os 
from transformers import LlamaTokenizer
import random
import pickle
import copy
import ipdb
from cn_tokenization import ChineseTokenizer

# class pretrain_dataset(dataset):
#     def __init__(self, args):
#         pass
#     def __item__():
#         pass

"""
def get_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_dict = {}
        for line in f:
            key, value = line.strip().split('\t')
            vocab_dict[key.strip()] = value.strip()
    return vocab_dict
"""
    


def reverse_index_list(_list:list, need_index:object) -> int:
    """
    Function:
        反向索引列表中固定元素
        逆序查找:在列表中查找指定元素的最后一个匹配项
    """
    try:
        return _list.__len__() - 1 - _list[::-1].index(need_index)
    except:
        return -1



def find_cut_token_point(token_list:list, vocab:dict) -> int:
    """
    Function:
        根据句号或逗号对应的tokenid， 寻找一个分割点, 此时
        根据['。', '，', '.']对应的token分别判断, 分别对应[511, 8024, 119]
    """
    for cuter_token in [vocab['。'], vocab['，'], vocab['.']]:
        cut_point = reverse_index_list(_list=token_list, need_index=cuter_token)
        if cut_point > 0:
            return cut_point



def split_token_by_maxlen(token_list:list, max_len:int, vocab:dict) -> list:
    """
    Function:
        根据最大长度，拆分文本，如果有句号，根据句号拆分，如果没有，根据逗号

        判断序列长度是否大于max_len,若小于,返回序列本身;若大于,则寻找截断点将序列截断。
        截断点的判断方法是在max_len长度范围内的最后一个中文句号对应的token,若没有中文句号则依次寻找最后一个逗号、最后一个英文句号
        若没有找到截断点,便根据max_len进行截断
        最后返回切分过的token序列
    Args:
        token_list -> zu
        max_len -> 最大长度限制
        vocab -> 词表
    Return:
        text_list -> 小于等于max_len的文本列表
    """
    if token_list.__len__() <= max_len:
        return [token_list]
    else:
        token_nested_lists = []
        while token_list.__len__() > max_len:
            cut_point = find_cut_token_point(token_list=token_list[:max_len], vocab=vocab)
            if cut_point:
                token_nested_lists.append(token_list[:cut_point])
                token_list = token_list[cut_point:]
            else:
                token_nested_lists.append(token_list[:max_len])
                token_list = token_list[max_len:]
        token_nested_lists.append(token_list)
        return token_nested_lists



def build_masked_language_task(seq_tokens:str, vocab:dict, vocab_token_ids_list:list) -> tuple:
    """
    Function:
        根据一个句子的token输入，构建mask任务
    Args:
        seq_tokens -> 句子的token
        vocab_token_ids_list -> 词表token得ind列表，用于替换mask
        vocab -> 词典
    Return:
        masked_seq_tokens -> 掩码后的句子的token
        mask_indexs -> 被掩码的位置的索引
    """
    lenght = seq_tokens.__len__()
    #mask掉一个序列中15%的部分
    masked_seq_num = round(lenght*0.15)
    if masked_seq_num == 0:
        masked_seq_num = 1
    mask_kinfe1 = round(masked_seq_num*0.8)
    mask_kinfe2 = round(masked_seq_num*0.9)
    mask_indexs = random.sample(range(lenght), k=masked_seq_num)
    masked_seq_tokens = copy.deepcopy(seq_tokens)

    # 替换为[MASK], 其token_id为103
    for ind in mask_indexs[:mask_kinfe1]:
        masked_seq_tokens[ind] = vocab['[MASK]']
    # 替换为随机词，并确保不会与[MASK]、[CLS]、[SEP]重叠，若重叠则把replace_token+1
    for ind in mask_indexs[mask_kinfe1:mask_kinfe2]:
        replace_token = random.choice(vocab_token_ids_list) 

        try:
            masked_seq_tokens[ind] = replace_token if replace_token not in [vocab['[MASK]'], vocab['[CLS]'], vocab['[SEP]']] else replace_token + 1
        except:
            ipdb.set_trace()
    # 不改变
    return masked_seq_tokens, mask_indexs



def build_pretrain_task(all_token_list:str, max_len:int, vocab_token_ids_list:list, vocab:dict) -> list:
    """
    Function:
        question -> 模型训练都是以token为单位，所以截断或mask都应该以token为单位
        根据单条文本，构建预训练文本，这里文本可能超过max_len的长度，
        对于超长文本，根据一个句号进行截断，生成多个文本
    Args:
        token_list -> 一段话经过tokenizer转化为token的列表
        max_len -> 最大长度限制
        vocab_token_ids_list -> vocab词典的index全部索引的列表
        vocab -> 词表
    Return:
        added_mlm_nsp_tokens -> 添加了mlm和nsp任务的预训练数据 
    """
    # 分几步分开进行
    # 1、将文本按最大长度进行截断
    # 2、将截断后的文本，拆分为上下句
    # 3、构建nsp任务 
    # 4、构建mask任务，可以分开构建

    less_max_len_token_list = []
    # 1、将超过最大长度的文本拆分切开
    for token_list in all_token_list:
        if len(token_list) == 0:
            pass
        else:
            token_nested_lists = split_token_by_maxlen(token_list=token_list, max_len=max_len, vocab=vocab)
            less_max_len_token_list.extend(token_nested_lists)

    # 4、合并上下句token，同时构建mlm任务
    # 对上下句创建随机掩码,生成掩码序列与掩码位置索引,并进行文本合并
    # 生成掩码后的上下句合并文本、未掩码的上下句合并文本、token_type_ids(0表示上句,1表示下句)、掩码位置索引、isnext
    added_mlm_nsp_tokens = []
    for token_list in less_max_len_token_list:
        masked_seq_tokens, mask_indexs = build_masked_language_task(
            seq_tokens=token_list, vocab=vocab, vocab_token_ids_list=vocab_token_ids_list
        )
        added_mlm_nsp_tokens.append([
            [vocab['[CLS]']] + masked_seq_tokens + [vocab['[SEP]']],
            [vocab['[CLS]']] + token_list + [vocab['[SEP]']],                                                   
            [i+1 for i in mask_indexs]                                      
        ])
    
    return added_mlm_nsp_tokens



def data_generator(dir_path:str, content_columns_name:str, tokenizer:ChineseTokenizer, max_len:int,) -> list:
    """
    Function:
        由于数据量较大，采用同文件路径下，分批次进行读取的方法，每次读取1个文件，进行数据处理，并返回
    Args:
        dir_path -> 数据所在的文件夹
        content_columns_name -> 文本的列名，因为以dataframe的形式存储，所以需要正文的列名
        tokenizer -> tokenizer
        max_len -> 接受的seq_len token最大长度
    Yield:
        added_mlm_nsp_tokens -> 添加过预训练任务的数据
    """
    # 
    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]
    saved_pretrain_data_dir = os.path.join(os.path.split(dir_path)[0], 'cache')
    if not os.path.exists(saved_pretrain_data_dir):
        os.mkdir(saved_pretrain_data_dir)
    vocab = tokenizer.get_vocab()
    vocab_token_ids_list = list(vocab.values())
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        full_file_path = os.path.join(dir_path, file_name)
        dataframe = read_parquet(full_file_path)
        all_tokens = tokenizer(dataframe[content_columns_name].values.tolist(), add_special_tokens=False)['input_ids']
        added_mlm_nsp_tokens = build_pretrain_task(
            all_token_list=all_tokens, max_len=max_len-3, vocab_token_ids_list=vocab_token_ids_list, vocab=vocab)

        yield added_mlm_nsp_tokens
    



if __name__ == "__main__":
    print('do')
    #tokenizer_path = '/home/daixingshuo/tokenizer/transformers_tokenizer/llama_chinese/chinese_llama.model'
    #tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer_path = '/home/daixingshuo/tokenizer/transformers_tokenizer/chinese/chinese.model'
    web_tokenizer = ChineseTokenizer(tokenizer_path)
    # 添加pad标记
    web_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    content_columns_name = 'content'
    dir_path = '/home/daixingshuo/pre_train_web/data/'
    max_len = 2048
    demo_generator = data_generator(
        dir_path=dir_path, content_columns_name=content_columns_name, tokenizer=web_tokenizer, max_len=max_len)
    for demo in demo_generator:
        print(1)
