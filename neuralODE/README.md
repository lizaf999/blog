# Neural Ordinary Differential Equations
NeurIPS 2018のBest PaperであるNeuralODEをフルスクラッチで実装しました．コードの詳細は[Neural Ordinary Differential Equationsの実装](https://segfault11.hatenablog.jp/entry/2020/08/05/120556)に書いてあります．

## Requirement
- Python 3.8.2
- Pytorch 1.5.0
- CUDA 10.1

で動作確認しています．PytorchやCUDAのバージョンを変えるとエラーを吐く可能性が高いです．

## Usage
```bash
python experiment.py
```
で学習が始まります．論文にも乗っているMNISTの実験です．パラメータ数は論文とは少し違います．

## License
This Project is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
