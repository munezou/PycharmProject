# insomnograf2-prototype-analyze

## ローカル環境での動作確認

### 前提
* `docker` 及び `docker-compose` が使用できること
* ローカル環境に `awslocal` がinstallされていること
```shell
pip install awscli-local
```

### 初期設定
```shell
make bootstrap
```

### 実行方法
AI_VERSIONには実行したいaiのversionを設定する。各数値がどのバージョンかは `ai_version.py` を参照。
```shell
AI_VERSION=1 make run
```

### 結果の確認
下記コマンドを実行すると解析結果が `tmp` フォルダ内に生成される。
```shell
make get_analyzed_result
```

### 解析したいedfを差し替える場合
`assets/edf/sample.edf` を解析するようになっている。このファイルを差し替えることで解析対象のedfの切り替えが可能。

### デバッグ
* `import pdb; pdb.set_trace()` を止めたい箇所に記述する`
* `docker-compose up -d local && docker attach isg-analyze_local` で解析を起動すると上記箇所で止まりデバッグできる

## ログ
### ローカル
`make run`を実行するとコンソールにそのままログが出る

### development
- ブラウザでログを漁る場合
    https://ap-northeast-1.console.aws.amazon.com/cloudwatch/home?region=ap-northeast-1#logsV2:log-groups/log-group/$252Fecs$252Fisg2-analyze
- リアルタイムにみたい場合
    ```shell
    aws logs --follow tail /ecs/isg2-analyze
    ```
