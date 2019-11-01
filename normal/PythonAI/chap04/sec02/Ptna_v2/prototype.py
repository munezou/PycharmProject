from ptna import *
""" 実行ブロック
"""

def prompt(obj):
    """ ピティナのプロンプトを作る関数
        戻り値 'Ptnaオブジェクト名:応答オブジェクト名 > '
    """
    return obj.get_name() + ':' + obj.get_responder_name() + '> '

print('Ptna System prototype : ptna') # プログラムの情報を表示
ptna = Ptna('ptna')                   # Ptnaオブジェクトを生成

while True:                           # 対話処理開始
    inputs = input(' > ')
    if not inputs:
        print('バイバイ')
        break
    response = ptna.dialogue(inputs)  # 応答文字列を取得
    print(prompt(ptna), response)     # プロンプトと応答文字列をつなげて表示
