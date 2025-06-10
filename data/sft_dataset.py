import json
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    """
    自定义监督微调 (SFT) 数据集类。
    从 JSONL 文件中读取每行包含 'query' 和 'answer' 的样本，
    并将其转换为符合聊天格式的字符串。
    """
    def __init__(self, file_path, tokenizer):
        """
        :param file_path: 包含 JSONL 数据的文件路径
        :param tokenizer: 分词器实例，用于后续模板生成
        """
        super().__init__()
        self.examples = self._load_data(file_path)  # 载入并解析所有样本
        self.tokenizer = tokenizer                 # 存储分词器实例

    @staticmethod
    def _load_data(file_path):
        """
        从 JSONL 文件逐行解析 JSON 对象。
        :param file_path: JSONL 文件路径
        :return: 样本列表，每项为 dict
        """
        items = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                items.append(json.loads(line))
        return items

    def __getitem__(self, index):
        """
        获取索引对应样本，构造 system/user/assistant 聊天格式字符串。
        :param index: 样本索引
        :return: 聊天格式文本，待 collate 进一步分词
        """
        ex = self.examples[index]
        dialog = [
            {'role': 'system',    'content': 'You are a helpful assistant.'},
            {'role': 'user',      'content': ex['query']},
            {'role': 'assistant', 'content': ex['answer']}
        ]
        # 返回未分词的对话字符串
        return self.tokenizer.apply_chat_template(dialog, tokenize=False)

    def __len__(self):
        """
        :return: 数据集样本数
        """
        return len(self.examples)
