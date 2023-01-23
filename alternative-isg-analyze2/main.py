import os
import sys
import traceback
import datetime
from datetime import date

import numpy as np

from ai_controller import AiController
from ai_version import AiVersion
from result import ScoringResult
from user import User
from constants import Constants
from gender import Gender


def configuration_parameter(edf_filename, rml_filename, user: User, ai_version: AiVersion):
    return [edf_filename, rml_filename, user, ai_version]


def to_list(element: ScoringResult) -> list:
    return [element.type_str, element.start_mills]


def _main() -> None:
    try:
        """
        setting value
        """
        # Analyze version
        ai_version = AiVersion.version1_2_4
        print(f"ai version: {ai_version.name}")
        
        # Patient Information
        scoring_result_id: str = '135138680'
        birthday: date = datetime.datetime(2022, 1, 1, 00, 00, 00)
        gender: Gender = Gender.Male
        
        # create user instance
        user: User = User(birthday=birthday, gender=gender, id=scoring_result_id, first_name='', last_name='')
        
        # extract edf file name.
        edf_dir = os.path.join(os.getcwd(), 'assets', 'edf')
        unique_edf_file_name = np.array([])
        for file in os.listdir(edf_dir):
            unique_edf_file_name = np.append(unique_edf_file_name, file)
        
        # extract rml file name
        rml_dir = os.path.join(os.getcwd(), 'assets', 'teacher_data')
        unique_rml_file_name = np.array([])
        for file in os.listdir(rml_dir):
            unique_rml_file_name = np.append(unique_rml_file_name, file)
        
        # Extract a valid edf file.
        valid_edf_file = np.array([])
        valid_rml_file = np.array([])
        for edf_file in unique_edf_file_name:
            base_edf, ext_edf = os.path.splitext(edf_file)
            for rml_file in unique_rml_file_name:
                base_rml, ext_rml = os.path.splitext(rml_file)
                if base_edf == base_rml:
                    valid_edf_file = np.append(valid_edf_file, edf_file)
                    valid_rml_file = np.append(valid_rml_file, rml_file)
            
        
        config_parameter = configuration_parameter(
            valid_edf_file[0], valid_rml_file[0], user, ai_version
        )

        ai_ctrl = AiController(config_parameter=config_parameter)
        ai_ctrl.start()
        
        lightsoff = ai_ctrl.lightsoff()
        lightson = ai_ctrl.lightson()

        results = ScoringResult.from_rml(os.path.join(os.getcwd(), 'output_file', Constants.RML_OUTPUT_DEFAULT_NAME))
        results = list(map(to_list, results))
        
    except Exception:
        print(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    _main()

    print("Successful analysis.")
