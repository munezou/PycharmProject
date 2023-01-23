import numpy as np
from scipy import signal

import pyedflib as edf

from constants import Constants
from common_ai import ret_eeg_idx
from arguments_return_values import PreProsessorReturn


class PreProcessor:
    def __init__(self):
        pass

    def start(self, edf_filenames: str):
        sampling_freq = Constants.SAMPLING_FREQUENCE

        edf_file = edf.EdfReader(edf_filenames)
        signal_labels = edf_file.getSignalLabels()
        eeg_idx = ret_eeg_idx(signal_labels)
        duration = edf_file.getFileDuration()

        fp1 = -edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["Fp1"]])
        fp2 = -edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["Fp2"]])
        a1 = -edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["A1"]])
        a2 = -edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["A2"]])

        lgt_off = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["Light"]])

        r_a1 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_A1"]])
        r_a2 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_A2"]])
        r_ref = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Ref"]])
        r_fp1 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Fp1"]])
        r_fp2 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Fp2"]])

        ma = (a1 + a2) / 2
        fp1_ma = fp1 - ma
        fp2_ma = fp2 - ma
        fp1_fp2 = fp1 - fp2
        m1_m2 = a1 - a2

        fp1_m2 = fp1 - a2
        fp2_m2 = fp2 - a2
        fp1_m1 = fp1 - a1
        fp2_m1 = fp2 - a1

        fp = np.array([0.3, 35])
        fs = np.array([0.15, 70])

        gpass = 3

        gstop = 5.5

        ffp2 = np.array([49.9, 50.1])
        ffs2 = np.array([45, 55])

        gpass2 = 3

        gstop2 = 40

        fp1 = self.bandpass(fp1, sampling_freq, fp, fs, gpass, gstop)
        fp2 = self.bandpass(fp2, sampling_freq, fp, fs, gpass, gstop)
        a1 = self.bandpass(a1, sampling_freq, fp, fs, gpass, gstop)
        a2 = self.bandpass(a2, sampling_freq, fp, fs, gpass, gstop)
        fp1_ma = self.bandpass(fp1_ma, sampling_freq, fp, fs, gpass, gstop)
        fp2_ma = self.bandpass(fp2_ma, sampling_freq, fp, fs, gpass, gstop)
        fp1_fp2 = self.bandpass(fp1_fp2, sampling_freq, fp, fs, gpass, gstop)
        m1_m2 = self.bandpass(m1_m2, sampling_freq, fp, fs, gpass, gstop)
        fp1_m1 = self.bandpass(fp1_m1, sampling_freq, fp, fs, gpass, gstop)
        fp2_m1 = self.bandpass(fp2_m1, sampling_freq, fp, fs, gpass, gstop)
        fp1_m2 = self.bandpass(fp1_m2, sampling_freq, fp, fs, gpass, gstop)
        fp2_m2 = self.bandpass(fp2_m2, sampling_freq, fp, fs, gpass, gstop)

        fp1 = self.bandstop(fp1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2 = self.bandstop(fp2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        a1 = self.bandstop(a1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        a2 = self.bandstop(a2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_ma = self.bandstop(fp1_ma, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_ma = self.bandstop(fp2_ma, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_fp2 = self.bandstop(fp1_fp2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        m1_m2 = self.bandstop(m1_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_m1 = self.bandstop(fp1_m1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_m1 = self.bandstop(fp2_m1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_m2 = self.bandstop(fp1_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_m2 = self.bandstop(fp2_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)

        num_of_epoch = int(len(fp1_ma) / (30 * sampling_freq))

        fp1_ma = fp1_ma[: num_of_epoch * 30 * sampling_freq]
        fp2_ma = fp2_ma[: num_of_epoch * 30 * sampling_freq]
        m1_m2 = m1_m2[: num_of_epoch * 30 * sampling_freq]

        fp1_fp2 = fp1_fp2[: num_of_epoch * 30 * sampling_freq]
        fp1_m1 = fp1_m1[: num_of_epoch * 30 * sampling_freq]
        fp2_m1 = fp2_m1[: num_of_epoch * 30 * sampling_freq]
        fp1_m2 = fp1_m2[: num_of_epoch * 30 * sampling_freq]
        fp2_m2 = fp2_m2[: num_of_epoch * 30 * sampling_freq]
        fp1 = fp1[: num_of_epoch * 30 * sampling_freq]
        fp2 = fp2[: num_of_epoch * 30 * sampling_freq]
        a1 = a1[: num_of_epoch * 30 * sampling_freq]
        a2 = a2[: num_of_epoch * 30 * sampling_freq]

        eeg = {
            "fp1": fp1,
            "fp2": fp2,
            "a1": a1,
            "a2": a2,
            "fp1_ma": fp1_ma,
            "fp2_ma": fp2_ma,
            "fp1_fp2": fp1_fp2,
            "m1_m2": m1_m2,
            "fp1_m1": fp1_m1,
            "fp2_m1": fp2_m1,
            "fp1_m2": fp1_m2,
            "fp2_m2": fp2_m2,
            "lgt_off": lgt_off,
            "r_a1": r_a1,
            "r_a2": r_a2,
            "r_ref": r_ref,
            "r_fp1": r_fp1,
            "r_fp2": r_fp2,
            "num_of_epoch": num_of_epoch,
        }

        lgt_off = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["Light"]])

        r_a1 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_A1"]])
        r_a2 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_A2"]])
        r_ref = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Ref"]])
        r_fp1 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Fp1"]])
        r_fp2 = edf_file.readSignal(eeg_idx[Constants.EEG_LABELS["R_Fp2"]])

        if (np.sort(np.unique(lgt_off)) != [0, 1]).all() is False:
            raise Exception("lgt_off contains something other than 0 and 1.")

        lightsoff = 0
        if lgt_off[0] == 1:
            lightsoff = 0
        else:
            lightsoff = lgt_off.tolist().index(1)

        if lgt_off[-1] != 0:
            lightson = len(lgt_off)
        else:
            lightson = len(lgt_off) - lgt_off.tolist()[::-1].index(1)

        print(f"lightsoff={lightsoff}, lightson={lightson}\n")

        lgt_off = lgt_off[29::30].reshape(-1, 1)
        r_a1 = r_a1[29::30].reshape(-1, 1)
        r_fp1 = r_fp1[29::30].reshape(-1, 1)
        r_ref = r_ref[29::30].reshape(-1, 1)
        r_fp2 = r_fp2[29::30].reshape(-1, 1)
        r_a2 = r_a2[29::30].reshape(-1, 1)

        elect_info = np.concatenate(
            [lgt_off, r_a1, r_fp1, r_ref, r_fp2, r_a2], axis=1
        )

        return PreProsessorReturn(
            eeg=eeg,
            light_off=lightsoff,
            light_on=lightson,
            elect_info=elect_info,
            duration=duration,
        )

    def bandpass(self, x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2
        wp = f_p / f_n
        ws = (
            f_s / f_n
        )

        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)

        b, a = signal.butter(nn, wn, "band")
        y = signal.filtfilt(b, a, x)
        return y

    def bandstop(self, x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2
        wp = f_p / f_n
        ws = (
            f_s / f_n
        )

        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)

        b, a = signal.butter(nn, wn, "bandstop")
        y = signal.filtfilt(b, a, x)
        return y
