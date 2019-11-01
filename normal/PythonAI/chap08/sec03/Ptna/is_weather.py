import requests

class WeatherResponder:
    def __init__(self):
    # ファイルを読み込み、地域の辞書オブジェクトを作成する
        with open('data/place_code.txt', 'r', encoding = 'utf_8'
                  ) as file:
            # 1行ずつ読み込んでリストにする
            lines = file.readlines()
        # 末尾の改行を取り除いた行データを保持するリスト
        new_lines = []
        # ファイルデータのリストから1行データを取り出す
        for line in lines:
            # 末尾の改行文字(\n)を取り除く
            line = line.rstrip('\n')
            # 空文字をチェック
            if (line!=''):
                # 空文字以外をリストnew_linesに追加
                new_lines.append(line)
        # 行データの単語とその意味を要素にするリスト
        separate = []
        # 末尾の改行を取り除いたリストから1行データを取り出す
        for line in new_lines:
            # タブで分割して質問と答えのリストを作る
            sp = line.split('\t')
            # リストseparateに追加する
            separate.append(sp)
        # 「地域名:id番号」のかたちで辞書オブジェクトにする
        self.place_code = dict(separate)

    def is_weather(self, place):
        """ 渡された地域名に対応するid番号を返すメソッド
        """
        if place in self.place_code:
            return self.get_weather(self.place_code[place])
        else:
            return('そこはわかんないよ〜')

    def get_weather(self, code):
        """  Weather Hacksに接続して天気予報を取得する
        """
        # Weather HacksのURL
        url = 'http://weather.livedoor.com/forecast/webservice/json/v1'
        # 'city'をキー、codeをその値とした辞書オブジェクトを作成
        payload = { 'city': code }
        # 天気予報の応答文字列
        weather_data = requests.get(url, params=payload).json()
        # 今日、明日、明後日の天気を順番に取り出す
        forecast = '天気予報だよ〜\n'
        for weather in weather_data['forecasts']:
            # 応答文字列を作成
            forecast += (
                '\n'                    # 改行
                + weather['dateLabel']  # 予報日を取得
                + 'の天気は'            # つなぎの文字列
                + weather['telop']      # 天気情報を取得
            )
        return forecast

