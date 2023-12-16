import os
import sys
import traceback
import datetime
from datetime import date
from csv import writer

import numpy as np

from ai_controller import AiController
from ai_version import AiVersion
from result import ScoringResult
from user import User
from constants import Constants
from gender import Gender


def configuration_parameter(
        edf_filename,
        rml_filename,
        user: User,
        ai_version: AiVersion,
        graphic_enable: bool,
        store_enable: bool
):
    return [
        edf_filename,
        rml_filename,
        user,
        ai_version,
        graphic_enable,
        store_enable
    ]


def to_list(element: ScoringResult) -> list:
    return [element.type_str, element.start_mills]


def list_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def delete_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")


def _main() -> None:
    try:
        """
        setting value
        """
        # Analyze version
        ai_version = AiVersion.version2_1_5
        print(f"ai version: {ai_version.name}")

        graphic_flag = True
        delete_flag = True
        store_flag = True

        """
        Delete image
        """
        if delete_flag:
            # Delete past calculation results.
            directory = os.path.join(os.getcwd(), 'output_file', 'image')
            file_paths = list_files(directory)
            delete_files(file_paths)

        # Patient Information
        scoring_result_id: str = '135138680'
        birthday: date = datetime.datetime(2022, 1, 1, 00, 00, 00)
        gender: Gender = Gender.Male
        
        # create user instance
        user: User = User(birthday=birthday, gender=gender, id=int(scoring_result_id), first_name='', last_name='')
        
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
        
        acc_ka_csv_path = os.path.join(os.getcwd(), 'output_file', f'{ai_version.name}_accuracy_kappa.csv')
        
        if not os.path.exists(acc_ka_csv_path):
            # prepare csv file.
            header_acc_kappa = ['ID', 'R_A1', 'R_Fp1', 'R_Fp2', 'R_A2', 'invalid num', 'accuracy', 'kappa']
            
            with open(acc_ka_csv_path, 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(header_acc_kappa)
                f_object.close()
        
        for i in range(unique_edf_file_name.shape[0]):
            config_parameter = configuration_parameter(
                valid_edf_file[i], valid_rml_file[i], user, ai_version, graphic_flag, store_flag
            )
            
            print(f"parse edf file: {valid_edf_file[i]}")
    
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
