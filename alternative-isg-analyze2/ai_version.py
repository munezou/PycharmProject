from enum import IntEnum
import re


class AiVersion(IntEnum):
    version1_1 = 0
    version1_2_0 = 1
    version1_2_1 = 2
    version1_2_2 = 3
    version1_2_3 = 4
    version1_2_4 = 5
    version2_0 = 6
    version2_1 = 7

    def is_ai_version1(self) -> bool:
        return bool(re.match(r"version1_", self.name))

    def is_ai_version1_1(self) -> bool:
        return bool(re.match(r"version1_1", self.name))

    def is_ai_version1_2(self) -> bool:
        return bool(re.match(r"version1_2", self.name))

    def is_ai_version1_2_0(self) -> bool:
        return bool(re.match(r"version1_2_0", self.name))

    def is_ai_version1_2_1(self) -> bool:
        return bool(re.match(r"version1_2_1", self.name))

    def is_ai_version1_2_2(self) -> bool:
        return bool(re.match(r"version1_2_2", self.name))

    def is_ai_version1_2_3(self) -> bool:
        return bool(re.match(r"version1_2_3", self.name))

    def is_ai_version1_2_4(self) -> bool:
        return bool(re.match(r"version1_2_4", self.name))

    def is_ai_version2(self) -> bool:
        return bool(re.match(r"version2_", self.name))

    def is_ai_version2_0(self) -> bool:
        return bool(re.match(r"version2_0", self.name))

    def is_ai_version2_1(self) -> bool:
        return bool(re.match(r"version2_1", self.name))
