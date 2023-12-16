import os
from datetime import timedelta
from datetime import date

import numpy as np

import pyedflib as edf

from user import User
from stage import Stage

from constants import Constants
from common_ai import ret_eeg_idx
from arguments_return_values import OutputGeneratorArguments


class OutputGenerator:
    __output_filenames: dict = dict()
    __user: User
    __date: date
    __duration: int
    __lightsoff: int
    __lightson: int
    __scoring_results: np.ndarray = None
    __eeg: dict = dict()
    __cps: list = list()
    __edffiltered_filenames: str
    __edf_filenames: str

    def __init__(self, output_generator_parameter: OutputGeneratorArguments) -> None:
        self.__output_filenames = output_generator_parameter.output_filenames
        self.__user = output_generator_parameter.user
        self.__scoring_results = output_generator_parameter.scoring_results
        self.__certainty = output_generator_parameter.certainty
        self.__duration = output_generator_parameter.duration
        self.__lightsoff = output_generator_parameter.lightsoff
        self.__lightson = output_generator_parameter.lightson
        self.__eeg = output_generator_parameter.eeg
        self.__edf_filenames = output_generator_parameter.edf_filenames
        self.__edffiltered_filenames = output_generator_parameter.edffilterd_name

    def copy_contents(self, output_file, input_file) -> None:
        line = input_file.readline()

        while line:
            output_file.write(line)
            line = input_file.readline()

    def copy_templete(self, output_file, input_filename) -> None:
        self.check_file_for_read(input_filename, 204, "txt")
        with open(input_filename) as r:
            self.copy_contents(output_file, r)

    def check_file_for_read(self, filename, code, extension) -> None:
        if os.path.exists(filename) == 0:
            raise Exception(f"This {filename} does not exist!")
        if os.path.splitext(filename)[1] != "." + extension:
            raise Exception(f"The extension of {filename} is incorrect!")
        buf_file = open(filename)
        buf_file.close()

    def check_file_for_write(self, filename, code) -> None:
        if os.path.exists(filename):
            buf_file = open(filename)
            buf_file.close()

    def make_rml_file(self) -> None:
        self.check_file_for_write(self.__output_filenames["rml_filename"], 202)

        with open(self.__output_filenames["rml_filename"], mode="w") as f:
            self.copy_templete(f, Constants.TEMPLATES["version"])
            f.write(self.patient_to_rml_style())
            self.copy_templete(f, Constants.TEMPLATES["channel_config"])
            f.write(self.aquisition_to_rml_style())
            f.write(self.scoring_start_rml_style())
            f.write(self.scoring_end_rml_style())
            self.copy_templete(f, Constants.TEMPLATES["option"])

    def results_to_certainty(self) -> list:
        certainty_string = []
        previous_date = self.__date - timedelta(seconds=30)

        for (s, c, i) in zip(self.__scoring_results, self.__certainty, range(len(self.__scoring_results))):
            previous_date = previous_date + timedelta(seconds=30)
            if s == 5:  # case of NS
                certainty_string.append(
                    f'{i + 1},{previous_date.strftime("%F %T")},NotScored,,,,,\n'
                )
            else:
                buffer = f'{i+1},{previous_date.strftime("%F %T")},{Stage(s).name}'
                for ce in c:
                    buffer += "," + "{:.4f}".format(ce)
                buffer += "\n"
                certainty_string.append(buffer)
        return certainty_string

    def make_certainty_file(self) -> None:
        # Check the file
        self.check_file_for_write(self.__output_filenames["certainty_filename"], 203)

        certainty_string = self.results_to_certainty()

        with open(self.__output_filenames["certainty_filename"], mode="w") as f:
            f.write("Epoch_#,Timestamp, Stage,Wake,REM,NonREM1,NonREM2,NonREM3\n")
            for cs in certainty_string:
                f.write(cs)

    def aquisition_to_rml_style(self):
        date = self.__date.strftime("%Y-%m-%dT%H:%M:%S")
        buffer = ""
        buffer += "   <Acquisition>\n"
        buffer += "      <AcqNumber>67</AcqNumber>\n"
        buffer += '      <Device DeviceType="EdfImport">\n'
        buffer += "         <SerialNumber>Edf-130.158.157.5</SerialNumber>\n"
        buffer += "      </Device>\n"
        buffer += "      <Sessions>\n"
        buffer += "         <Session>\n"
        buffer += "            <RecordingStart>" + date + "</RecordingStart>\n"
        buffer += "            <Duration>" + str(self.__duration) + "</Duration>\n"
        buffer += "            <LightsOff>" + str(self.__lightsoff) + "</LightsOff>\n"
        buffer += "            <LightsOn>" + str(self.__lightson) + "</LightsOn>\n"
        buffer += "            <Segments>\n"
        buffer += "               <Segment>\n"
        buffer += "                  <StartTime>" + date + "</StartTime>\n"
        buffer += "                  <Duration>" + str(self.__duration) + "</Duration>\n"
        buffer += "               </Segment>\n"
        buffer += "            </Segments>\n"
        buffer += "         </Session>\n"
        buffer += "      </Sessions>\n"
        buffer += "   </Acquisition>\n"
        buffer += "   <CustomEventTypeDefs>\n"
        buffer += '      <ce:CustomEventTypeDef Name="BRX" Type="CustomDuration" Color="DarkTurquoise">\n'
        buffer += '         <ce:LogicalInfo Family="Neuro" />\n'
        buffer += "      </ce:CustomEventTypeDef>\n"
        buffer += "   </CustomEventTypeDefs>\n"
        buffer += "   <AcquisitionCommentDefs />\n"

        return buffer

    def scoring_result_to_change_point(self):
        previous_stage = -1
        change_points = []
        for (s, i) in zip(self.__scoring_results, range(len(self.__scoring_results))):
            if s != previous_stage:
                change_points.append({"start": i * 30, "label": Stage(s).name})
                previous_stage = s
        self.__cps = change_points

    def patient_to_rml_style(self):
        buffer = ""
        buffer += '   <Patient PatientType="Adult">\n'
        buffer += "      <PatientID>ZWVlKsbPC8s=</PatientID>\n"
        buffer += "      <ContactInfo>\n"
        buffer += "         <FirstName>ZWVlKsbPC8s=</FirstName>\n"
        buffer += "         <LastName>ZWVlKsbPC8s=</LastName>\n"
        buffer += "         <MiddleName />\n"
        buffer += "         <Address />\n"
        buffer += "         <Address2 />\n"
        buffer += "         <City />\n"
        buffer += "         <State />\n"
        buffer += "         <ZipCode />\n"
        buffer += "         <Phone />\n"
        buffer += "         <Phone2 />\n"
        buffer += "         <Fax />\n"
        buffer += "         <Email />\n"
        buffer += "      </ContactInfo>\n"
        buffer += "      <BirthDate>/93mRoP5uppcUKaXVpnm/A==</BirthDate>\n"
        buffer += f"      <Gender>{str(self.__user.gender.name)}</Gender>\n"
        buffer += "   </Patient>\n"

        return buffer

    def scoring_start_rml_style(self):
        rml_event_strings = ""
        rml_event_strings += "   <ScoringData>\n"
        rml_event_strings += "      <LastModified>2018-01-15T14:06:41</LastModified>\n"
        rml_event_strings += "      <Events>\n"
        rml_event_strings += "      </Events>\n"

        return rml_event_strings

    def scoring_end_rml_style(self):
        self.scoring_result_to_change_point()
        return self.change_point_to_rml_style()

    def change_point_to_rml_style(self):
        rml_stage_strings = ""
        rml_stage_strings += '      <StagingData StagingDisplay="NeuroAdultAASM">\n'
        rml_stage_strings += "         <UserStaging>\n"
        rml_stage_strings += "            <NeuroAdultAASMStaging>\n"
        for cp in self.__cps:
            rml_stage_strings += (
                '               <Stage Type="'
                + cp["label"]
                + '" Start="'
                + str(cp["start"])
                + '"/>\n'
            )
        rml_stage_strings += "            </NeuroAdultAASMStaging>\n"
        rml_stage_strings += "         </UserStaging>\n"
        rml_stage_strings += "      </StagingData>\n"
        rml_stage_strings += "   </ScoringData>\n"

        return rml_stage_strings

    def generate_filtered_edf(self) -> None:
        sampling_freq = Constants.SAMPLING_FREQUENCE

        edf_file = edf.EdfReader(self.__edf_filenames)
        signal_labels = edf_file.getSignalLabels()
        eeg_idx = ret_eeg_idx(signal_labels)

        edf_w = edf.EdfWriter(
            self.__edffiltered_filenames, len(Constants.ALL_SIGNAL_NAME), file_type=edf.FILETYPE_EDF
        )

        # Header writing (channal common part)
        edf_w.setAdmincode(edf_file.getAdmincode())
        edf_w.setBirthdate(self.__user.birthday)
        edf_w.setEquipment(edf_file.getEquipment())
        edf_w.setGender(self.__user.gender.value)
        edf_w.setPatientAdditional(edf_file.getPatientAdditional())
        edf_w.setPatientCode(str(self.__user.id))
        edf_w.setPatientName("anonymous")
        edf_w.setStartdatetime(edf_file.getStartdatetime())
        edf_w.setTechnician("anonymous")
        self.__date = edf_file.getStartdatetime()

        # Writing header (ch separate part)
        for i, name in enumerate(Constants.ALL_SIGNAL_NAME):
            edf_w.setLabel(i, name)

        signal_digital_maximum = edf_file.getDigitalMaximum(eeg_idx[Constants.EEG_LABELS["A1"]])
        signal_digital_minimum = edf_file.getDigitalMinimum(eeg_idx[Constants.EEG_LABELS["A1"]])

        signal_physical_maximum = edf_file.getPhysicalMaximum(eeg_idx[Constants.EEG_LABELS["A1"]])
        signal_physical_minimum = edf_file.getPhysicalMinimum(eeg_idx[Constants.EEG_LABELS["A1"]])

        signal_physical_dimension = edf_file.getPhysicalDimension(eeg_idx[Constants.EEG_LABELS["A1"]])

        for i in range(len(Constants.BRAIN_SIGNAL_NAME)):
            edf_w.setDigitalMaximum(i, signal_digital_maximum)
            edf_w.setDigitalMinimum(i, signal_digital_minimum)
            edf_w.setPhysicalDimension(i, signal_physical_dimension)
            edf_w.setPhysicalMaximum(i, signal_physical_maximum)
            edf_w.setPhysicalMinimum(i, signal_physical_minimum)
            edf_w.setPrefilter(i, "HP:0.3Hz,LP:35Hz")
            edf_w.setTransducer(i, "trans1")
            edf_w.setSamplefrequency(i, sampling_freq)

        light_digital_maximum = edf_file.getDigitalMaximum(eeg_idx[Constants.EEG_LABELS["Light"]])
        light_digital_minimum = edf_file.getDigitalMinimum(eeg_idx[Constants.EEG_LABELS["Light"]])

        light_physical_maximum = edf_file.getPhysicalMaximum(eeg_idx[Constants.EEG_LABELS["Light"]])
        light_physical_minimum = edf_file.getPhysicalMinimum(eeg_idx[Constants.EEG_LABELS["Light"]])

        light_physical_dimension = edf_file.getPhysicalDimension(
            eeg_idx[Constants.EEG_LABELS["Light"]]
        )
        light_sampling_freq = edf_file.getSampleFrequency(eeg_idx[Constants.EEG_LABELS["Light"]])

        light_off_index = Constants.ALL_SIGNAL_NAME.index("Light_OFF")
        edf_w.setDigitalMaximum(light_off_index, light_digital_maximum)
        edf_w.setDigitalMinimum(light_off_index, light_digital_minimum)
        edf_w.setPhysicalDimension(light_off_index, light_physical_dimension)
        edf_w.setPhysicalMaximum(light_off_index, light_physical_maximum)
        edf_w.setPhysicalMinimum(light_off_index, light_physical_minimum)
        edf_w.setPrefilter(light_off_index, "None")
        edf_w.setTransducer(light_off_index, "trans1")
        edf_w.setSamplefrequency(light_off_index, light_sampling_freq)

        regist_digital_maximum = edf_file.getDigitalMaximum(eeg_idx[Constants.EEG_LABELS["R_A1"]])
        regist_digital_minimum = edf_file.getDigitalMinimum(eeg_idx[Constants.EEG_LABELS["R_A1"]])

        regist_physical_maximum = edf_file.getPhysicalMaximum(eeg_idx[Constants.EEG_LABELS["R_A1"]])
        regist_physical_minimum = edf_file.getPhysicalMinimum(eeg_idx[Constants.EEG_LABELS["R_A1"]])

        regist_physical_dimension = edf_file.getPhysicalDimension(
            eeg_idx[Constants.EEG_LABELS["R_A1"]]
        )
        regist_sampling_freq = edf_file.getSampleFrequency(eeg_idx[Constants.EEG_LABELS["R_A1"]])

        for i in range(len(Constants.BRAIN_SIGNAL_NAME) + 1, len(Constants.ALL_SIGNAL_NAME)):
            edf_w.setDigitalMaximum(i, regist_digital_maximum)
            edf_w.setDigitalMinimum(i, regist_digital_minimum)
            edf_w.setPhysicalDimension(i, regist_physical_dimension)
            edf_w.setPhysicalMaximum(i, regist_physical_maximum)
            edf_w.setPhysicalMinimum(i, regist_physical_minimum)
            edf_w.setPrefilter(i, "None")
            edf_w.setTransducer(i, "trans1")
            edf_w.setSamplefrequency(i, regist_sampling_freq)

        edf_w.update_header()

        light_sampling_freq = int(light_sampling_freq)
        regist_sampling_freq = int(regist_sampling_freq)
        
        for i in range(int(self.__duration)):
            edf_w.writeSamples(
                [
                    np.ascontiguousarray(self.__eeg["fp1_ma"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["fp2_ma"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["fp1_fp2"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["m1_m2"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["fp1"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["fp2"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["a1"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(self.__eeg["a2"][i * sampling_freq: (i + 1) * sampling_freq]),
                    np.ascontiguousarray(
                        self.__eeg["lgt_off"][i * light_sampling_freq: (i + 1) * light_sampling_freq]
                    ),
                    np.ascontiguousarray(
                        self.__eeg["r_a1"][i * regist_sampling_freq: (i + 1) * regist_sampling_freq]
                    ),
                    np.ascontiguousarray(
                        self.__eeg["r_a2"][i * regist_sampling_freq: (i + 1) * regist_sampling_freq]
                    ),
                    np.ascontiguousarray(
                        self.__eeg["r_ref"][i * regist_sampling_freq: (i + 1) * regist_sampling_freq]
                    ),
                    np.ascontiguousarray(
                        self.__eeg["r_fp1"][i * regist_sampling_freq: (i + 1) * regist_sampling_freq]
                    ),
                    np.ascontiguousarray(
                        self.__eeg["r_fp2"][i * regist_sampling_freq: (i + 1) * regist_sampling_freq]
                    ),
                ]
            )
        
        edf_file._close()
        del edf_file

        edf_w.close()
        del edf_w

        edf_file = edf.EdfReader(self.__edffiltered_filenames)
        header = edf_file.getHeader()
        edf_file._close()
        del edf_file
