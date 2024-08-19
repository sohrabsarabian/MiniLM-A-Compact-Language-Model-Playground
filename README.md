# MiniLM: A Compact Language Model Playground 🚀🧠

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to MiniLM, your gateway to the fascinating world of compact language models! This project offers a streamlined implementation of a small-scale Language Model (LM) with only 19 million parameters. It's designed to allowing you to explore the inner workings of transformer-based language models without the need for extensive computational resources.

## 🌟 Features

- **Compact Design**: A mere 19 million parameters, trainable on a single GPU.
- **Modular Architecture**: Clean, object-oriented code structure for easy understanding and modification.
- **Customizable Training**: Adjust key parameters like batch size, context length, and model architecture via command-line arguments.
- **Interactive Sampling**: Generate text samples on-the-fly to observe the model's learning progress.
- **Performance Tracking**: Integration with Weights & Biases for detailed training metrics and visualizations.

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/sohrabsarabian/MiniLM-A-Compact-Language-Model-Playground.git
   cd MiniLM
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the following files in your project directory:
   - `wiki.txt`: Your training data
   - `wiki_trained_tokenizer.model`: Pre-trained SentencePiece tokenizer model
   - `wiki_trained_tokenizer.vocab`: Vocabulary file for the tokenizer

📝 Note on Data Files


Due to file size limitations on GitHub, the wiki.txt file is not included in this repository. Please download it separately in the Installation section. This file is necessary for training the model.

## 🚀 Usage

Train the model with default parameters:
```
python main.py
```

Customize training with command-line arguments:
```
python main.py --batch_size 16 --context 1024 --embed_size 512 --n_layers 8 --n_heads 8 --lr 1e-4 --train_iters 200000 --wandb
```

## 📊 Model Architecture

MiniLM is based on the transformer architecture and includes:

- Multi-head self-attention mechanisms
- Feed-forward neural networks
- Layer normalization
- Dropout for regularization

The model is designed to be easily understood and modified, making it an excellent starting point for those looking to dive deeper into the world of language models.

## 📈 Training and Evaluation

The training process includes:

- Batch-wise training on the wiki dataset
- Periodic evaluation on a held-out validation set
- Dynamic learning rate adjustment using a cosine annealing schedule
- Gradient clipping to prevent exploding gradients

Progress is logged using Weights & Biases, allowing you to track loss, learning rate, and generated samples throughout the training process.

## 🧪 Experiments to Try

1. Adjust the model size (layers, heads, embedding dimension) and observe the impact on performance and training speed.
2. Experiment with different learning rates and batch sizes to find the optimal configuration.
3. Try fine-tuning the model on different datasets to see how it adapts to various domains.
4. Implement and compare different attention mechanisms or positional encoding strategies.

## 🤝 Contributing

I welcome contributions to MiniLM! Whether it's bug fixes, feature additions, or documentation improvements, your input is valuable. Please feel free to submit pull requests or open issues for discussion.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- This project draws inspiration from the broader field of transformer-based language models, including the groundbreaking work on GPT and BERT.
- Special thanks to the PyTorch and Hugging Face communities for their invaluable resources and tools.

---

Embark on your journey into the world of language models with MiniLM. Happy coding and happy learning! 🎉📚
