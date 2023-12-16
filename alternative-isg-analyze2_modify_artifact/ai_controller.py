import codecs
import os
from datetime import datetime, timedelta
import traceback
import ntpath
import shutil
from csv import writer

import pyedflib as edf
import numpy as np
from IPython.display import display

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score
from xml.dom import minidom
import pandas as pd

from ai_version import AiVersion
from output_generator import OutputGenerator
from user import User
from analyzer import Analyzer
from pre_processor import PreProcessor
from post_processor import PostProcessor
from constants import Constants, InvalidElectrodeStatus
from arguments_return_values import PostProcessorArguments
from arguments_return_values import AnalyzerArguments, OutputGeneratorArguments
from common_ai import Electrode


class AiController:
    __ai_version: AiVersion
    __edf_file_names: str
    __rml_file_names: str
    __user: User
    __edffilter_names = os.path.join(
        os.getcwd(), 'output_file', Constants.EDF_AFTER_FILTER_DEFAULT_NAME
    )
    __rml_output_default_name = os.path.join(
        os.getcwd(), 'output_file', Constants.RML_OUTPUT_DEFAULT_NAME
    )
    __certainty_output_default_name = os.path.join(
        os.getcwd(), 'output_file', Constants.CERTAINTY_OUTPUT_DEFAULT_NAME
    )
    __output_filenames: dict = dict()
    __lightsoff: int
    __lightson: int
    __default_edf_filename: str = "template.edf"
    __clinic_tech_results: np.ndarray = np.array([])
    __electrode_instance: Electrode = Electrode()
    __graphic_enable: bool
    __store_enable: bool
    
    try:
        def __init__(self, config_parameter) -> None:
            if len(config_parameter) != 6:
                raise Exception("Syntax error: # of Input != 6")
    
            self.__edf_file_names = os.path.join(os.getcwd(), 'assets', 'edf', config_parameter[0])
            self.__rml_file_names = os.path.join(os.getcwd(), 'assets', 'teacher_data', config_parameter[1])
            self.__user = config_parameter[2]
            self.__ai_version = config_parameter[3]
            self.__graphic_enable = config_parameter[4]
            self.__store_enable = config_parameter[5]
    
        def start(self) -> None:
            self.load_check_args()
    
            preprocessor_result = PreProcessor().start(
                self.__edf_file_names, self.__graphic_enable, self.__store_enable
            )
    
            self.__lightsoff = preprocessor_result.light_off
            self.__lightson = preprocessor_result.light_on

            electrode_np = np.sum(preprocessor_result.elect_info[:, 1:6] >= 100, axis=0)
            elctrode_ratio = (electrode_np / preprocessor_result.elect_info.shape[0])

            if elctrode_ratio[0] <= 0.1:
                self.__electrode_instance.ai_analyze_status_a1 = InvalidElectrodeStatus.analyzed
            if elctrode_ratio[1] <= 0.1:
                self.__electrode_instance.ai_analyze_status_fp1 = InvalidElectrodeStatus.analyzed
            if elctrode_ratio[3] <= 0.1:
                self.__electrode_instance.ai_analyze_status_fp2 = InvalidElectrodeStatus.analyzed
            if elctrode_ratio[4] <= 0.1:
                self.__electrode_instance.ai_analyze_status_a2 = InvalidElectrodeStatus.analyzed


            analyzer_parameter = AnalyzerArguments(
                eeg=preprocessor_result.eeg,
                ai_version=self.__ai_version,
                elect_info=preprocessor_result.elect_info,
            )
    
            certainty, scoring_result = Analyzer(
                ai_version=self.__ai_version, arguments=analyzer_parameter
            ).start()
            
            preprocessor_result.eeg["m1_m2"] = preprocessor_result.chin
            
            post_processor_parameter = PostProcessorArguments(
                ai_version=self.__ai_version,
                certainty=certainty,
                scoring_result=scoring_result,
                light_on_epoch=int(self.__lightson / 30)
            )

            scoring_result = PostProcessor(
                post_processor_parameter=post_processor_parameter
            ).start()
            
            # rml file
            self.__clinic_tech_results = self.from_rml()
            
            basename_without_ext = os.path.splitext(os.path.basename(self.__edf_file_names))[0]
            
            clinic_tech_parse_stage_path = os.path.join(
                os.getcwd(),
                'output_file',
                'results',
                f'{basename_without_ext}_clinic_pase_stage.csv'
            )
            
            df_clinic_tech_results = pd.DataFrame(self.__clinic_tech_results)

            if self.__store_enable:
                df_clinic_tech_results.to_csv(clinic_tech_parse_stage_path)
            
            stage_clinical_tec = np.array([], dtype='uint16')
            
            for time, stage in self.__clinic_tech_results:
                if stage == 'Wake':
                    stage_clinical_tec = np.append(stage_clinical_tec, 0)
                elif stage == 'REM':
                    stage_clinical_tec = np.append(stage_clinical_tec, 1)
                elif stage == 'NonREM1':
                    stage_clinical_tec = np.append(stage_clinical_tec, 2)
                elif stage == 'NonREM2':
                    stage_clinical_tec = np.append(stage_clinical_tec, 3)
                elif stage == 'NonREM3':
                    stage_clinical_tec = np.append(stage_clinical_tec, 4)
                elif stage == 'NotScored':
                    stage_clinical_tec = np.append(stage_clinical_tec, 5)
                else:
                    raise ValueError('No supported stage!')
            
            scoring_result_str = np.array([])
            for stage_num in scoring_result:
                if stage_num == 0:
                    scoring_result_str = np.append(scoring_result_str, 'Wake')
                elif stage_num == 1:
                    scoring_result_str = np.append(scoring_result_str, 'REM')
                elif stage_num == 2:
                    scoring_result_str = np.append(scoring_result_str, 'NonREM1')
                elif stage_num == 3:
                    scoring_result_str = np.append(scoring_result_str, 'NonREM2')
                elif stage_num == 4:
                    scoring_result_str = np.append(scoring_result_str, 'NonREM3')
                elif stage_num == 5:
                    scoring_result_str = np.append(scoring_result_str, 'NotScored')
                else:
                    raise ValueError('No supported stage!')
            
            # confirm stage length
            if len(scoring_result) < len(self.__clinic_tech_results):
                stage_clinical_tec_str = self.__clinic_tech_results[:, 1]
                stage_clinical_tec = np.delete(stage_clinical_tec_str, -1)
            
            # prepare data without 'NotScored'
            scoring_result_str_without_not_scored = np.array([])
            stage_clinical_tec_without_not_scored = np.array([])
            
            stage_index = 0
            # Ignore NotScored.
            for data_stage in stage_clinical_tec:
                # confirm scoring_result value
                confirm_scoring_data = scoring_result_str[stage_index]
                if data_stage == 'NotScored' or confirm_scoring_data == 'NotScored':
                    print(f"which stage either is NotScored(EPOCH:{stage_index + 1})")
                else:
                    scoring_result_str_without_not_scored = np.append(
                        scoring_result_str_without_not_scored,
                        scoring_result_str[stage_index]
                    )
                    
                    stage_clinical_tec_without_not_scored = np.append(
                        stage_clinical_tec_without_not_scored,
                        stage_clinical_tec[stage_index]
                    )
                
                stage_index += 1
            
            scoring_result_str.tofile('scoring_result.csv', sep=',')
            stage_clinical_tec_str.tofile('clinic_tec.csv', sep=',')
            
            print()

            arg_np = np.sum(preprocessor_result.elect_info[:, 1:6] >= 100, axis=0)

            arg_np = np.concatenate([arg_np[0:2], arg_np[3:5]])

            ratio = (arg_np / preprocessor_result.elect_info.shape[0])
            print(f"ratio: R_A1={ratio[0]}, R_Fp1={ratio[1]}, R_Fp2={ratio[2]}, R_A2={ratio[3]}")

            invalid_number_of_electrodes = 0
            
            for data in ratio:
                if data >= 0.1:
                    invalid_number_of_electrodes += 1
            
            # calculate accuracy
            accuracy = accuracy_score(stage_clinical_tec_without_not_scored, scoring_result_str_without_not_scored)
            print(f"accuracy: {accuracy}\n")
            
            # calculate confusion matrix
            matrix = confusion_matrix(
                stage_clinical_tec_without_not_scored,
                scoring_result_str_without_not_scored,
                labels=["Wake", "REM", "NonREM1", "NonREM2", "NonREM3"],
                normalize='true'
            )

            pd.set_option("display.max_columns", 10)
            pd.set_option("display.max_rows", 10)
            
            cm = self.make_cm(matrix, ["Wake", "REM", "NonREM1", "NonREM2", "NonREM3"])
            print("           predict")
            display(cm)

            # prepare data
            base_name, ext_rml = os.path.splitext(self.__edf_file_names)
            edf_id = ntpath.basename(base_name)
            
            #
            confusion_matrix_path = os.path.join(
                os.getcwd(), 'output_file', f'{self.__ai_version.name}_confusion_matrix.csv'
            )
            
            for i, data in enumerate(matrix):
                if i == 0:
                    print(f"{edf_id},,,,,", file=codecs.open(confusion_matrix_path, 'a', 'utf-8'))
                    
                    print(
                        ", Wake, REM, NonREM1, NonREM2, NonREM3",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
                    
                    print(
                        f"Wake, {matrix[i][0]}, {matrix[i][1]}, {matrix[i][2]}, {matrix[i][3]}, {matrix[i][4]}",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
                elif i == 1:
                    print(
                        f"REM, {matrix[i][0]}, {matrix[i][1]}, {matrix[i][2]}, {matrix[i][3]}, {matrix[i][4]}",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
                elif i == 2:
                    print(
                        f"NonREM1, {matrix[i][0]}, {matrix[i][1]}, {matrix[i][2]}, {matrix[i][3]}, {matrix[i][4]}",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
                elif i == 3:
                    print(
                        f"NonREM2, {matrix[i][0]}, {matrix[i][1]}, {matrix[i][2]}, {matrix[i][3]}, {matrix[i][4]}",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
                elif i == 4:
                    print(
                        f"NonREM3, {matrix[i][0]}, {matrix[i][1]}, {matrix[i][2]}, {matrix[i][3]}, {matrix[i][4]}",
                        file=codecs.open(confusion_matrix_path, 'a', 'utf-8')
                    )
            
            print()
            print()
            
            # calculate kappa static
            k = cohen_kappa_score(
                stage_clinical_tec_without_not_scored,
                scoring_result_str_without_not_scored,
                weights='quadratic'
            )
            print(f"weighted kappa k: {k}\n\n")
            
            acc_ka_list = [
                edf_id,
                ratio[0],
                ratio[1],
                ratio[2],
                ratio[3],
                invalid_number_of_electrodes,
                accuracy,
                k
            ]

            with open(
                    os.path.join(os.getcwd(), 'output_file', f'{self.__ai_version.name}_accuracy_kappa.csv'),
                    'a',
                    newline=''
            ) as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(acc_ka_list)
                f_object.close()

            preprocessor_result.eeg["m1_m2"] = preprocessor_result.chin
            
            if (
                int(Constants.SAMPLING_FREQUENCE * preprocessor_result.duration) >
                preprocessor_result.eeg['fp1'].shape[0]
            ):
                preprocessor_result.duration = (
                    int(preprocessor_result.eeg['fp1'].shape[0] / Constants.SAMPLING_FREQUENCE)
                )
            
            output_generator_parameter = OutputGeneratorArguments(
                output_filenames=self.__output_filenames,
                user=self.__user,
                scoring_results=scoring_result,
                certainty=certainty,
                duration=preprocessor_result.duration,
                lightsoff=self.__lightsoff,
                lightson=self.__lightson,
                eeg=preprocessor_result.eeg,
                edf_filenames=self.__edf_file_names,
                edffilterd_name=self.__edffilter_names,
            )
    
            output_generator = OutputGenerator(
                output_generator_parameter=output_generator_parameter
            )
    
            output_generator.generate_filtered_edf()
            output_generator.make_rml_file()
            output_generator.make_certainty_file()
            
            if self.__store_enable:
                base_file = os.path.splitext(os.path.basename(self.__edf_file_names))[0]
                destination_file = f'{base_file}_edf_filtered.edf'
                destination_path = os.path.join(os.getcwd(), 'output_file', 'results', destination_file)
                shutil.copyfile(self.__edffilter_names, destination_path)
    
                destination_file = f'{base_file}_rml.rml'
                destination_path = os.path.join(os.getcwd(), 'output_file', 'results', destination_file)
                shutil.copyfile(self.__rml_output_default_name, destination_path)
    
                destination_file = f'{base_file}_result.csv'
                destination_path = os.path.join(os.getcwd(), 'output_file', 'results', destination_file)
                shutil.copyfile(self.__certainty_output_default_name, destination_path)
    
        def check_edf(self) -> None:
            edf_file = edf.EdfReader(self.__edf_file_names)
            default_edf_file = edf.EdfReader(self.__default_edf_filename)
            edf_file_signal_headers = edf_file.getSignalHeaders()
            default_edf_file_signal_headers = default_edf_file.getSignalHeaders()
    
            if edf_file.getEquipment() != default_edf_file.getEquipment():
                raise Exception("EDF Format Error")
            if len(edf_file_signal_headers) != len(default_edf_file_signal_headers):
                raise Exception("EDF Format Error")
    
            signal_headers_compare_list = list(
                zip(edf_file_signal_headers, default_edf_file_signal_headers)
            )
    
            for signal_headers in signal_headers_compare_list:
                if signal_headers[0].keys() != signal_headers[1].keys():
                    raise Exception("EDF Format Error")
    
                for k, v in signal_headers[0].items():
                    if (v not in ["Fp1", "Fp2", "A1", "A2"]) and (
                        k
                        in [
                            "physical_max",
                            "physical_min",
                            "digital_max",
                            "digital_min",
                        ]
                    ):
                        continue
                    if signal_headers[0][k] != signal_headers[1][k]:
                        raise Exception("EDF Format Error")
    
        def load_check_args(self) -> None:
            self.check_edf()
    
            self.__output_filenames = {
                "edffiltered_filename": self.__edffilter_names,
                "rml_filename": self.__rml_output_default_name,
                "certainty_filename": self.__certainty_output_default_name,
            }
        
        def from_rml(self) -> np.ndarray:
            xml = minidom.parse(self.__rml_file_names)
            stages = xml.getElementsByTagName('Stage')
            start_time = xml.getElementsByTagName("RecordingStart")[0]
            during_time = xml.getElementsByTagName("Duration")[0]
            start_time = start_time.firstChild.data
            start_time = start_time.replace('T', ' ')
            start_time_object = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            during_time = during_time.firstChild.data
            during_time_int = int(during_time)
            finish_time_object = start_time_object + during_time_int * timedelta(seconds=1)
            past_delta_time: int = 0
            past_t: int = 0
            time = np.array([])
            stage = np.array([])
            results = np.array([])
            for s in stages:
                t = s.attributes['Type'].value
                start = s.attributes['Start'].value
                delta_time = int(start)
                if delta_time != 0:
                    during_epoc = int((delta_time - past_delta_time) / 30)
                    
                    for i in range(during_epoc):
                        if i != 0:
                            input_time = past_delta_time + i * 30
                            measuring_time = start_time_object + input_time * timedelta(seconds=1)
                            time = np.append(time, str(measuring_time))
                            stage = np.append(stage, past_t)
                
                measuring_time = start_time_object + delta_time * timedelta(seconds=1)
                time = np.append(time, str(measuring_time))
                stage = np.append(stage, t)
                past_measuring_time = measuring_time
                past_t = t
                past_delta_time = delta_time
            
            if finish_time_object > past_measuring_time:
                num_epoc_fill = int((during_time_int - past_delta_time) / 30)
                for j in range(num_epoc_fill + 1):
                    if j != 0:
                        input_time = past_delta_time + j * 30
                        measuring_time = start_time_object + input_time * timedelta(seconds=1)
                        time = np.append(time, str(measuring_time))
                        stage = np.append(stage, past_t)
            
            # Join columns
            results = np.c_[time, stage]
            
            return results
        
        def make_cm(self, matrix, columns):
            n = len(columns)
            
            act = ['Real data'] * n
            pred = ['Predict data'] * n
            
            cm = pd.DataFrame(
                matrix,
                #columns=[pred, columns],
                columns=[columns],
                #index=[act, columns]
                index=[columns]
            )
            
            return cm
    
    except Exception:
        print(traceback.format_exc())
    except ValueError as e:
        print(e)
    


    def ai_version(self) -> AiVersion:
        return self.__ai_version

    def output_filenames(self) -> dict:
        return self.__output_filenames

    def lightsoff(self) -> int:
        return self.__lightsoff

    def lightson(self) -> int:
        return self.__lightson

    def electrode_status(self) -> Electrode:
        return self.__electrode_instance
