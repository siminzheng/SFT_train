# SFT_train
以Meta-Llama-3.1-8B-Instruct为基座模型，手动实现一个简单的SFT监督微调

目录架构为：
```text
llama-lora-finetune/
├── README.md
├── requirements.txt
├── configs/
│   └── training_args.py
├── data/
│   ├── preprocessing.py
│   └── sft_dataset.py
├── models/
│   └── loader.py
├── utils/
│   ├── logging.py
│   └── collate.py
└── scripts/
    ├── train.py
    └── sft_train.py
```
