# python library
from abc import ABCMeta, abstractmethod
import pandas as pd


class Users(metaclass=ABCMeta):
    """

    """
    @abstractmethod
    def get_name(self, user_id: int) -> str:
        pass

    @abstractmethod
    def put_name(self, user_id: int, name: str):
        pass

    def get_family_name(self, user_id) -> str:
        return self.get_name(user_id).split()[0]


class DictUsers(Users):
    def __init__(self, d: dict):
        self._d = d

    def get_name(self, user_id: int) -> str:
        return self._d[user_id]

    def put_name(self, user_id: int, name: str):
        self._d[user_id] = name

    def get_dict(self):
        return self._d


class DataFrameUsers(Users):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_name(self, user_id: int) -> str:
        return self._df.loc[user_id, "name"]

    def put_name(self, user_id: int, name: str):
        self._df.loc[user_id, "name"] = name


def main() -> None:
    # using dict
    print("-------------< using dict >-----------------------")
    dict_users = DictUsers({1: "田中 はじめ", 2: "梅田 よしお"})

    print(issubclass(dict_users.__class__, Users))  # True
    print(isinstance(dict_users, Users))  # True

    print(dict_users.get_name(1))  # 田中 はじめ
    print(dict_users.get_dict())  # {1: '田中 はじめ', 2: '梅田 よしお'}

    print(f"{dict_users.get_family_name(2)}\n")  # 梅田

    # using pandas
    print("-------------< using pandas >-----------------------")
    df_users = DataFrameUsers(pd.DataFrame(
        {"name": ["田中 はじめ", "梅田 よしお"]},
        index=[1, 2]
    ))

    print(df_users.get_name(1))  # 田中 はじめ

    df_users.put_name(2, "高木 よしお")
    print(df_users.get_family_name(2))  # 高木


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    main()
