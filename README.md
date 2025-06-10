# SFT_train
以Meta-Llama-3.1-8B-Instruct为基座模型，通过lora手动实现一个简单的SFT监督微调

目录架构为：
```text
SFT_train/
├── README.md
├── requirements.txt
├── data/
│   └── sft_dataset.py
├── utils/
│   └── collate.py
└── scripts/
    └── sft_train.py
```

```bash
pip install -r requirements.txt
python scripts/sft_train.py
```
