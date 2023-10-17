from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from functools import partial

from transformers.configuration_utils import PretrainedConfig

from cn_tokenization import ChineseTokenizer
from tokenizer import data_generator
from data import WebDataset, data_collator
from model import WebEncoderForPreTraining


def train(config, model, optimizer, train_dataloaders,):
    #设置初始loss与train_step
    logging_loss = 0
    train_step = 0
    LOSS = 1e8
    #循环从1开始到epochs+1结束(为什么从1开始？)
    for epoch in range(1, config.train.epochs+1):
        #每个epoch对train_dataloaders进行迭代
        for train_data in train_dataloaders:
            """
            DataLoader类实现在训练中批量加载数据
            dataset:数据集对象
            batch_size:批次大小
            shuffle:每个epoch是否对数据进行洗牌
            collate_fn:自定义收集函数，本质是对每个批次的数据进行预处理
            """
            train_dataset = WebDataset(data=train_data)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=partial(data_collator, max_len=config.tokenizer.max_len))
            #enumerate获取批次索引与批次内容
            for ind, (input_ids, attention_mask, masked_lm_labels) in enumerate(train_dataloader):
                print(ind)
                train_step += 1
                #数据移动到gpu上
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                masked_lm_labels = masked_lm_labels.to(config.device)
                #next_sentence_label = next_sentence_label.to(config.device)
                #数据传入模型，计算损失
                loss = model(input_ids, attention_mask, masked_lm_labels)
                logging_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #每隔100个step打印并记录一次训练信息
                if train_step % config.train.logging_step == 0:
                    logger.info(f'Epoch: {epoch}, Step: {train_step}, Loss = {logging_loss/config.logging_step:.6f}')
                    with open(config.train.writer_dir, mode='w') as f:
                        f.write(f'Epoch: {epoch}, Step: {train_step}, Loss = {logging_loss/config.logging_step:.6f}' + '\n')
                    logging_loss = 0


if __name__ == "__main__":
    configdir = '/home/daixingshuo/pre_train_web/config.yaml'
    conf = OmegaConf.load(configdir)
    model_conf = PretrainedConfig.from_dict(conf.model)

    tokenizer_path = '/home/daixingshuo/tokenizer/transformers_tokenizer/chinese/chinese.model'
    web_tokenizer = ChineseTokenizer(tokenizer_path)

    #获取词汇表大小
    conf.vocab_size = web_tokenizer.vocab_size
    model = WebEncoderForPreTraining(model_conf).to(conf.device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.learning_rate)
    dir_path = conf.data.pretrain_data_dir
    content_columns_name = conf.data.content_columns_name
    max_len = conf.tokenizer.max_len
    train_dataloaders = data_generator(
        dir_path=dir_path, content_columns_name=content_columns_name, tokenizer=web_tokenizer, max_len=max_len
        )
    train(config=conf, model=model, optimizer=optimizer, train_dataloaders=train_dataloaders)