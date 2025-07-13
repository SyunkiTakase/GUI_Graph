# GUI_Graph
Metric Learningのサンプルコード

## ファイル＆フォルダ一覧

<details>
<summary>フォルダ</summary>
 
|ファイル名|説明|
|----|----|
|Config|学習用のハイパーパラメータが記載されたConfigファイルが格納されたフォルダ．|
|fig|README用の画像を保存するフォルダ．|
|output|学習結果のログやモデルを保存するフォルダ．|
</details>

<details>
<summary>ファイル</summary>
 
|ファイル名|説明|
|----|----|
|plot_gui_graph.py|GUI上でCSVファイルを読み込んで，グラフを可視化するコード．|
|train.py|ResNet-18を学習するコード．|
|trainer.py|学習ループのコード．|


|ファイル名|説明|
|----|----|
|Config/resnet_config.py|ResNet-18用のハイパーパラメータが定義されたコード．|

</details>

## 実行手順

### 学習
ハイパーパラメータ等は，Configフォルダ内の各ファイルで設定してください．

* ResNet-18のファインチューニング(CIFAR-10)
```
$ python train.py --config_path ./Config/resnet_config.py
```

### 描画

* 描画ツールの起動
```
python plot_gui_graph.py
```
![START](./fig/02.png)

