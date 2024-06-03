from datetime import date

from dateutil.parser import parse

from gender import Gender


class User:
    id: int
    full_name: str
    birthday: date
    gender: Gender

    DEFAULT_BIRTHDAY = "2222-01-01"

    def __init__(
        self,
        birthday: date,
        gender: int,
        id: int,
        last_name: str,
        first_name: str,
    ):
        self.id = id
        self.full_name = f"{last_name} {first_name}"
        self.birthday = parse(self.DEFAULT_BIRTHDAY) if birthday is None else birthday

        # EDFではMale/FemaleしかうけつけないがdatabaseにはAnotherがある
        # AnotherはEDF上はMaleとして扱う
        if gender == 1:
            self.gender = Gender.Female
        else:
            self.gender = Gender.Male
