import torch

def sft_collate(batch, tokenizer, end_str, max_length):
    """
    自定义 collate 函数：
    1. 将 batch 中的聊天模板文本进行分词、截断与填充
    2. 基于 end_str 构建 loss_mask，使得只对回答部分计算损失

    :param batch: 未分词的文本列表
    :param tokenizer: 分词器实例
    :param end_str: 回答开始标志字符串
    :param max_length: 最大序列长度
    :return: inputs(dict of tensors), loss_mask(tensor)
    """
    # 强制使用固定结束标志
    end_str = "<|start_header_id|>assistant<|end_header_id""|>\n\n"

    # 分词、填充、截断
    inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True)
    input_ids = inputs['input_ids']
    seq_len = len(input_ids[0])

    # 获取 end_str 对应的 token id 列表
    end_ids = tokenizer(end_str)['input_ids']
    e_len = len(end_ids)

    loss_mask = []
    for ids in input_ids:
        # 搜索 end_ids 在 ids 中最后出现的位置
        for i in range(len(ids) - e_len, -1, -1):
            if ids[i:i + e_len] == end_ids:
                # 回答部分置 1，之前部分置 0
                mask = [1] * (seq_len - 1)
                mask[:i + e_len - 1] = [0] * (i + e_len - 1)
                loss_mask.append(mask)
                break
            if i == 0:
                # 若未找到结束标志，则全部置 0
                loss_mask.append([0] * (seq_len - 1))

    # 转为 PyTorch 张量
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    loss_mask = torch.tensor(loss_mask)
    return inputs, loss_mask
