import requests
from requests.auth import HTTPBasicAuth

from device import Device
from inspection import Inspection
from inspection_result import InspectionResult
from scoring_result import ScoringResult2
from user import User
from ai_version import AiVersion


class Slack:
    SLACK_DEV_GROUP_ID = "S014ZR5EG48"
    SLACK_BIZDEV_GROUP_ID = "S014ZR7C7ME"
    SLACK_REQUEST_SERIALIZE_API = (
        "https://ba3cbezwu2.execute-api.us-east-1.amazonaws.com/prod/slack_request"
    )
    BASIC_AUTH_ID = "cae5=8t!!+cjp5e-jtase"
    BASIC_AUTH_PASSWORD = "@:rsl+.--sgksg_kxu=hh"

    def __init__(
        self,
        user: User = None,
        device: Device = None,
        inspection: Inspection = None,
        inspection_result: InspectionResult = None,
        scoring_result: ScoringResult2 = None,
    ):
        self.user = user
        self.device = device
        self.inspection = inspection
        self.inspection_result = inspection_result
        self.scoring_result = scoring_result

    def notify_start_analysis(
        self, suimin_ts: str, ai_version: AiVersion, color: str = None
    ) -> str:
        if suimin_ts:
            return self.__common_append_message(
                new_color="warning",
                new_pretext=f"=> Analysis started {AiVersion(ai_version).name}",
                suimin_ts=suimin_ts,
            )
        else:
            return self.__common_post_message(
                color="warning",
                pretext=f"Analysis started {AiVersion(ai_version).name}",
            )

    def notify_finish_analysis(self, suimin_ts: str):
        self.__common_append_message(
            new_color="good", new_pretext="=> Analysis complete", suimin_ts=suimin_ts
        )

    def notify_analysis_error(self, suimin_ts: str):
        self.__common_append_message(
            new_color="danger",
            new_pretext=f" => Analysis failed <!subteam^{self.SLACK_DEV_GROUP_ID}>",
            suimin_ts=suimin_ts,
        )

    def __common_post_message(self, pretext: str, color: str):
        data = {
            "command": "post",
            "channel": self.__general_notification_channel(),
            "arguments": {
                "color": color,
                "pretext": pretext,
                "fields": [
                    self.__common_user_field(),
                    self.__common_inspection_field(),
                    self.__common_inspection_result_field(),
                    self.__common_scoring_result_field(),
                ],
            },
        }

        response = requests.post(
            Slack.SLACK_REQUEST_SERIALIZE_API,
            timeout=10,
            json=data,
            auth=HTTPBasicAuth(Slack.BASIC_AUTH_ID, Slack.BASIC_AUTH_PASSWORD),
        )

        if response.status_code != 200:
            message = response.json()
            print("Slack request serializer api post call error")
            print(response.status_code)
            print(message)
        else:
            return response.json()["message"]["suimin_ts"]

    def __common_update_message(self, target_response):
        return self.client.chat_update(
            channel=target_response["channel"],
            ts=target_response["ts"],
            attachments=target_response["message"]["attachments"],
        )

    def __common_append_message(
        self, suimin_ts: str, new_color: str = None, new_pretext: str = None
    ):
        data = {
            "command": "append",
            "suimin_ts": suimin_ts,
            "arguments": {"new_color": new_color, "new_pretext": new_pretext},
        }
        response = requests.post(
            Slack.SLACK_REQUEST_SERIALIZE_API,
            timeout=10,
            json=data,
            auth=HTTPBasicAuth(Slack.BASIC_AUTH_ID, Slack.BASIC_AUTH_PASSWORD),
        )

        if response.status_code != 200:
            message = response.json()
            print("Slack request serializer api append call error")
            print(response.status_code)
            print(message)
            return
        else:
            return response.json()["message"]["suimin_ts"]

    def __general_notification_channel(self):
        return "#notify-isg2-test" if self.device.demo is True else "#notify-isg2"

    def __common_user_field(self):
        return {
            "title": "ユーザー",
            "value": f"id: {self.user.id} {self.user.full_name}",
            "short": True,
        }

    def __common_inspection_field(self):
        return {"title": "検査", "value": f"id: {self.inspection.id}", "short": True}

    def __common_inspection_result_field(self):
        return {
            "title": "検査結果",
            "value": f"id: {self.inspection_result.id}",
            "short": True,
        }

    def __common_scoring_result_field(self):
        return {
            "title": "解析結果",
            "value": f"id: {self.scoring_result.id}",
            "short": True,
        }
