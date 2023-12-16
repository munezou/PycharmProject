import os.path

import numpy as np
import copy
import gc
from scipy import signal

import pyedflib as edf
import joblib

from constants import Constants
from common_ai import ret_eeg_idx
from arguments_return_values import PreProsessorReturn, TimeFrequencyReturn
from fft_function import ov, hanning, fft_ave, plot_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import emd
import matplotlib.pyplot as plt
import pandas as pd

# plot imf raw data
def imf_raw_data_plot(
        index_name: str,
        t,
        signal_data_IMF1,
        signal_data_IMF2,
        signal_data_IMF3,
        signal_data_IMF4,
        signal_data_IMF5,
        signal_data_IMF6,
        signal_data_IMF7,
        signal_name: str
):
    # imf raw
    fig = plt.figure(figsize=(25, 20))
    ax0 = fig.add_subplot(7, 1, 1)
    ax1 = fig.add_subplot(7, 1, 2)
    ax2 = fig.add_subplot(7, 1, 3)
    ax3 = fig.add_subplot(7, 1, 4)
    ax4 = fig.add_subplot(7, 1, 5)
    ax5 = fig.add_subplot(7, 1, 6)
    ax6 = fig.add_subplot(7, 1, 7)

    for i in range(7):
        if not i:
            ax0.plot(t, signal_data_IMF1, color="red", label=f"{signal_name} IMF1 raw")
            ax0.set_ylim(-1, 1)
            ax0.legend()
        elif i == 1:
            ax1.plot(t, signal_data_IMF2, color="darkgreen", label=f"{signal_name} IMF2 raw")
            ax1.set_ylim(-1, 1)
            ax1.legend()
        elif i == 2:
            ax2.plot(t, signal_data_IMF3, color="m", label=f"{signal_name} IMF3 raw")
            ax2.set_ylim(-1, 1)
            ax2.legend()
        elif i == 3:
            ax3.plot(t, signal_data_IMF4, color="darkorange", label=f"{signal_name} IMF4 raw")
            ax3.set_ylim(-1, 1)
            ax3.legend()
        elif i == 4:
            ax4.plot(t, signal_data_IMF5, color="red", label=f"{signal_name} IMF5 raw")
            ax4.set_ylim(-1, 1)
            ax4.legend()
        elif i == 5:
            ax5.plot(t, signal_data_IMF6, color="darkgreen", label=f"{signal_name} IMF6 raw")
            ax5.set_ylim(-1, 1)
            ax5.legend()
        elif i == 6:
            ax6.plot(t, signal_data_IMF7, color="m", label=f"{signal_name} IMF7 raw")
            ax6.set_ylim(-1, 1)
            ax6.legend()

    fig.tight_layout()
    plot_path = os.path.join(
        os.getcwd(), 'output_file', 'image', 'emd', f'{index_name}_{signal_name}_imf_artifact_raw.png'
    )
    plt.savefig(plot_path)
    # plt.show()
    plt.close()

    del fig
    gc.collect()


def time_frequency_compare_plot(
        epoch_index,
        signal_name,
        samplerate,
        t1,
        time_imf1,
        time_imf2,
        time_imf3,
        time_imf4,
        time_imf5,
        time_imf6,
        time_imf7,
        fft_axis_imf1,
        fft_mean_imf1,
        fft_axis_imf2,
        fft_mean_imf2,
        fft_axis_imf3,
        fft_mean_imf3,
        fft_axis_imf4,
        fft_mean_imf4,
        fft_axis_imf5,
        fft_mean_imf5,
        fft_axis_imf6,
        fft_mean_imf6,
        fft_axis_imf7,
        fft_mean_imf7,
):
    # imf raw
    fig = plt.figure(figsize=(25, 20))
    ax0 = fig.add_subplot(14, 2, 1)  # IMF1 time
    ax1 = fig.add_subplot(14, 2, 2)  # IMF1 freq
    ax2 = fig.add_subplot(14, 2, 3)  # IMF2 time
    ax3 = fig.add_subplot(14, 2, 4)  # IMF2 freq
    ax4 = fig.add_subplot(14, 2, 5)  # IMF3 time
    ax5 = fig.add_subplot(14, 2, 6)  # IMF3 freq
    ax6 = fig.add_subplot(14, 2, 7)  # IMF4 time
    ax7 = fig.add_subplot(14, 2, 8)  # IMF4 freq
    ax8 = fig.add_subplot(14, 2, 9)  # IMF5 time
    ax9 = fig.add_subplot(14, 2, 10)  # IMF5 freq
    ax10 = fig.add_subplot(14, 2, 11)  # IMF6 time
    ax11 = fig.add_subplot(14, 2, 12)  # IMF6 freq
    ax12 = fig.add_subplot(14, 2, 13)  # IMF7 time
    ax13 = fig.add_subplot(14, 2, 14)  # IMF7 freq

    for i in range(14):
        if not i:
            ax0.plot(t1, time_imf1, color="red", label=f"{signal_name} IMF1 time")
            ax0.set_ylim(-0.5, 0.5)
            ax0.legend()
        elif i == 1:
            fft_mean_max = np.abs(np.max(fft_mean_imf1))
            fft_mean_min = np.abs(np.min(fft_mean_imf1))
            if fft_mean_max >= fft_mean_min:
                ax1_yRange = fft_mean_max
            else:
                ax1_yRange = fft_mean_min
            ax1.set_xticks(np.arange(0, samplerate, 50))
            ax1.set_yticks(np.arange(0, 3, 0.5))
            ax1.set_xlim(0, 200)
            ax1.set_ylim(0, ax1_yRange)
            ax1.plot(fft_axis_imf1, fft_mean_imf1, color="red", label=f"{signal_name} IMF1 frequency", lw=1)
        elif i == 2:
            ax2.plot(t1, time_imf2, color="darkgreen", label=f"{signal_name} IMF2 time")
            ax2.set_ylim(-0.5, 0.5)
            ax2.legend()
        elif i == 3:
            fft_mean_max = np.abs(np.max(fft_mean_imf2))
            fft_mean_min = np.abs(np.min(fft_mean_imf2))
            if fft_mean_max >= fft_mean_min:
                ax3_yRange = fft_mean_max
            else:
                ax3_yRange = fft_mean_min
            ax3.set_xticks(np.arange(0, samplerate, 50))
            ax3.set_yticks(np.arange(0, 3, 0.5))
            ax3.set_xlim(0, 200)
            ax3.set_ylim(0, ax3_yRange)
            ax3.plot(fft_axis_imf2, fft_mean_imf2, color="darkgreen", label=f"{signal_name} IMF2 frequency", lw=1)
        elif i == 4:
            ax4.plot(t1, time_imf3, color="m", label=f"{signal_name} IMF3 time")
            ax4.set_ylim(-0.5, 0.5)
            ax4.legend()
        elif i == 5:
            fft_mean_max = np.abs(np.max(fft_mean_imf3))
            fft_mean_min = np.abs(np.min(fft_mean_imf3))
            if fft_mean_max >= fft_mean_min:
                ax5_yRange = fft_mean_max
            else:
                ax5_yRange = fft_mean_min
            ax5.set_xticks(np.arange(0, samplerate, 50))
            ax5.set_yticks(np.arange(0, 3, 0.5))
            ax5.set_xlim(0, 200)
            ax5.set_ylim(0, ax5_yRange)
            ax5.plot(fft_axis_imf3, fft_mean_imf3, color="m", label=f"{signal_name} IMF3 frequency", lw=1)
        elif i == 6:
            ax6.plot(t1, time_imf4, color="darkorange", label=f"{signal_name} IMF4 time")
            ax6.set_ylim(-0.5, 0.5)
            ax6.legend()
        elif i == 7:
            fft_mean_max = np.abs(np.max(fft_mean_imf4))
            fft_mean_min = np.abs(np.min(fft_mean_imf4))
            if fft_mean_max >= fft_mean_min:
                ax7_yRange = fft_mean_max
            else:
                ax7_yRange = fft_mean_min
            ax7.set_xticks(np.arange(0, samplerate, 50))
            ax7.set_yticks(np.arange(0, 3, 0.5))
            ax7.set_xlim(0, 200)
            ax7.set_ylim(0, ax7_yRange)
            ax7.plot(fft_axis_imf4, fft_mean_imf4, color="darkorange", label=f"{signal_name} IMF4 frequency", lw=1)
        elif i == 8:
            ax8.plot(t1, time_imf5, color="red", label=f"{signal_name} IMF5 time")
            ax8.set_ylim(-0.5, 0.5)
            ax8.legend()
        elif i == 9:
            fft_mean_max = np.abs(np.max(fft_mean_imf5))
            fft_mean_min = np.abs(np.min(fft_mean_imf5))
            if fft_mean_max >= fft_mean_min:
                ax9_yRange = fft_mean_max
            else:
                ax9_yRange = fft_mean_min
            ax9.set_xticks(np.arange(0, samplerate, 50))
            ax9.set_yticks(np.arange(0, 3, 0.5))
            ax9.set_xlim(0, 200)
            ax9.set_ylim(0, ax9_yRange)
            ax9.plot(fft_axis_imf5, fft_mean_imf5, color="red", label=f"{signal_name} IMF5 frequency", lw=1)
        elif i == 10:
            ax10.plot(t1, time_imf6, color="darkgreen", label=f"{signal_name} IMF6 time")
            ax10.set_ylim(-0.5, 0.5)
            ax10.legend()
        elif i == 11:
            fft_mean_max = np.abs(np.max(fft_mean_imf6))
            fft_mean_min = np.abs(np.min(fft_mean_imf6))
            if fft_mean_max >= fft_mean_min:
                ax11_yRange = fft_mean_max
            else:
                ax11_yRange = fft_mean_min
            ax11.set_xticks(np.arange(0, samplerate, 50))
            ax11.set_yticks(np.arange(0, 3, 0.5))
            ax11.set_xlim(0, 200)
            ax11.set_ylim(0, ax11_yRange)
            ax11.plot(fft_axis_imf6, fft_mean_imf6, color="darkgreen", label=f"{signal_name} IMF6 frequency", lw=1)
        elif i == 12:
            ax12.plot(t1, time_imf7, color="m", label=f"{signal_name} IMF7 time")
            ax12.set_ylim(-0.5, 0.5)
            ax12.legend()
        elif i == 13:
            fft_mean_max = np.abs(np.max(fft_mean_imf6))
            fft_mean_min = np.abs(np.min(fft_mean_imf6))
            if fft_mean_max >= fft_mean_min:
                ax13_yRange = fft_mean_max
            else:
                ax13_yRange = fft_mean_min
            ax13.set_xticks(np.arange(0, samplerate, 50))
            ax13.set_yticks(np.arange(0, 3, 0.5))
            ax13.set_xlim(0, 200)
            ax13.set_ylim(0, ax13_yRange)
            ax13.plot(fft_axis_imf7, fft_mean_imf7, color="darkgreen", label=f"{signal_name} IMF7 frequency", lw=1)

    fig.tight_layout()
    plot_path = os.path.join(
        os.getcwd(), 'output_file', 'image', 'summary',
        f'{epoch_index}_{signal_name}_imf_compare_artifact.png'
    )
    plt.savefig(plot_path)
    # plt.show()
    plt.close()

    del fig
    gc.collect()


def transform_from_time_to_frequency_domain(
        training_data: pd.DataFrame,
        epoch: str,
        signal_name: str,
        t1: np.ndarray,
        samplerate: float,
        Fs: int,
        overlap: int,
        t2: np.ndarray,
        graphic_enable: bool,
) -> TimeFrequencyReturn:
    """
    ------------------------------------------
    fp1_m2 signal
    ------------------------------------------
    """
    # divide IMF data >-------------------------------
    time_IMF1 = training_data.loc[epoch, f'{signal_name}_IMF1_0':f'{signal_name}_IMF1_5999']
    time_IMF2 = training_data.loc[epoch, f'{signal_name}_IMF2_0':f'{signal_name}_IMF2_5999']
    time_IMF3 = training_data.loc[epoch, f'{signal_name}_IMF3_0':f'{signal_name}_IMF3_5999']
    time_IMF4 = training_data.loc[epoch, f'{signal_name}_IMF4_0':f'{signal_name}_IMF4_5999']
    time_IMF5 = training_data.loc[epoch, f'{signal_name}_IMF5_0':f'{signal_name}_IMF5_5999']
    time_IMF6 = training_data.loc[epoch, f'{signal_name}_IMF6_0':f'{signal_name}_IMF6_5999']
    time_IMF7 = training_data.loc[epoch, f'{signal_name}_IMF7_0':f'{signal_name}_IMF7_5999']

    # plot IMF raw
    if graphic_enable:
        imf_raw_data_plot(
            epoch,
            t1,
            time_IMF1.values,
            time_IMF2.values,
            time_IMF3.values,
            time_IMF4.values,
            time_IMF5.values,
            time_IMF6.values,
            time_IMF7.values,
            signal_name,
        )

    # averaged FFT >-------------------
    # IMF1
    time_array, N_ave = ov(time_IMF1.values, samplerate, Fs, overlap)
    time_array_imf1, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf1, fft_axis_imf1 = fft_ave(time_array_imf1, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF1',
            samplerate,
            N_ave,
            t2,
            time_array_imf1,
            fft_axis_imf1,
            fft_mean_imf1
        )

    # IMF2
    time_array, N_ave = ov(time_IMF2.values, samplerate, Fs, overlap)
    time_array_imf2, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf2, fft_axis_imf2 = fft_ave(time_array_imf2, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF2',
            samplerate,
            N_ave,
            t2,
            time_array_imf2,
            fft_axis_imf2,
            fft_mean_imf2
        )

    # IMF3
    time_array, N_ave = ov(time_IMF3.values, samplerate, Fs, overlap)
    time_array_imf3, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf3, fft_axis_imf3 = fft_ave(time_array_imf3, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF3',
            samplerate,
            N_ave,
            t2,
            time_array_imf3,
            fft_axis_imf3,
            fft_mean_imf3
        )

    # IMF4
    time_array, N_ave = ov(time_IMF4.values, samplerate, Fs, overlap)
    time_array_imf4, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf4, fft_axis_imf4 = fft_ave(time_array_imf4, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF4',
            samplerate,
            N_ave,
            t2,
            time_array_imf4,
            fft_axis_imf4,
            fft_mean_imf4
        )

    # IMF5
    time_array, N_ave = ov(time_IMF5.values, samplerate, Fs, overlap)
    time_array_imf5, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf5, fft_axis_imf5 = fft_ave(time_array_imf5, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF5',
            samplerate,
            N_ave,
            t2,
            time_array_imf5,
            fft_axis_imf5,
            fft_mean_imf5
        )

    # IMF6
    time_array, N_ave = ov(time_IMF6.values, samplerate, Fs, overlap)
    time_array_imf6, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf6, fft_axis_imf6 = fft_ave(time_array_imf6, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF6',
            samplerate,
            N_ave,
            t2,
            time_array_imf6,
            fft_axis_imf6,
            fft_mean_imf6
        )

    # IMF7
    time_array, N_ave = ov(time_IMF7.values, samplerate, Fs, overlap)
    time_array_imf7, acf = hanning(time_array, Fs, N_ave)
    fft_array, fft_mean_imf7, fft_axis_imf7 = fft_ave(time_array_imf7, samplerate, Fs, N_ave, acf)

    if graphic_enable:
        plot_data(
            epoch,
            f'{signal_name}_IMF7',
            samplerate,
            N_ave,
            t2,
            time_array_imf7,
            fft_axis_imf7,
            fft_mean_imf7
        )

    if graphic_enable:
        time_frequency_compare_plot(
            epoch,
            signal_name,
            samplerate,
            t1,
            time_IMF1.values,
            time_IMF2.values,
            time_IMF3.values,
            time_IMF4.values,
            time_IMF5.values,
            time_IMF6.values,
            time_IMF7.values,
            fft_axis_imf1,
            fft_mean_imf1,
            fft_axis_imf2,
            fft_mean_imf2,
            fft_axis_imf3,
            fft_mean_imf3,
            fft_axis_imf4,
            fft_mean_imf4,
            fft_axis_imf5,
            fft_mean_imf5,
            fft_axis_imf6,
            fft_mean_imf6,
            fft_axis_imf7,
            fft_mean_imf7,
        )

    return TimeFrequencyReturn(
        imf1_ndarray=fft_mean_imf1,
        imf2_ndarray=fft_mean_imf2,
        imf3_ndarray=fft_mean_imf3,
        imf4_ndarray=fft_mean_imf4,
        imf5_ndarray=fft_mean_imf5,
        imf6_ndarray=fft_mean_imf6,
        imf7_ndarray=fft_mean_imf7
    )


class PreProcessor:
    def __init__(self):
        pass

    def start(self, edf_filenames: str, graphic_enable: bool, store_enable: bool):
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

        # common signal
        chin = copy.copy(a1 - a2)

        # Signal preparation for artifact detection
        fp1_m2_detection = copy.copy(fp1_m2)
        fp2_m1_detection = copy.copy(fp2_m1)
        fp1_ma_detection = copy.copy(fp1_ma)
        fp2_ma_detection = copy.copy(fp2_ma)
        fp1_detection = copy.copy(fp1)
        fp2_detection = copy.copy(fp2)
        a1_detection = copy.copy(a1)
        a2_detection = copy.copy(a2)
        fp1_fp2_detection = copy.copy(fp1_fp2)
        m1_m2_detection = copy.copy(m1_m2)
        chin_detection = copy.copy(chin)

        # band-pass filter on normal eeg
        fp = np.array([0.3, 35])
        fs = np.array([0.15, 70])
        gpass = 3
        gstop = 5.5

        # high-pass filter on detection
        fp_high = np.array([0.3])
        fs_high = np.array([0.15])
        gpass_high = 3
        gstop_high = 5.5

        # band-pass filter at chin
        fp_chin = np.array([0.3, 70])
        fs_chin = np.array([0.15, 90])
        gpass_chin = 3
        gstop_chin = 5.5

        # common band-stop filter
        ffp2 = np.array([49.9, 50.1])
        ffs2 = np.array([45, 55])
        gpass2 = 3
        gstop2 = 40
        ffp3 = np.array([59.9, 60.1])
        ffs3 = np.array([55, 65])

        """
        ----------------------
        eeg normal processing
        ----------------------
        """
        fp1 = self.bandpass(fp1, sampling_freq, fp, fs, gpass, gstop)
        fp2 = self.bandpass(fp2, sampling_freq, fp, fs, gpass, gstop)
        a1 = self.bandpass(a1, sampling_freq, fp, fs, gpass, gstop)
        a2 = self.bandpass(a2, sampling_freq, fp, fs, gpass, gstop)
        fp1_ma = self.bandpass(fp1_ma, sampling_freq, fp, fs, gpass, gstop)
        fp2_ma = self.bandpass(fp2_ma, sampling_freq, fp, fs, gpass, gstop)
        fp1_fp2 = self.bandpass(fp1_fp2, sampling_freq, fp, fs, gpass, gstop)
        chin = self.bandpass(chin, sampling_freq, fp_chin, fs_chin, gpass_chin, gstop_chin)
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
        chin = self.bandstop(chin, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        m1_m2 = self.bandstop(m1_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_m1 = self.bandstop(fp1_m1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_m1 = self.bandstop(fp2_m1, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_m2 = self.bandstop(fp1_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_m2 = self.bandstop(fp2_m2, sampling_freq, ffp2, ffs2, gpass2, gstop2)

        fp1 = self.bandstop(fp1, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2 = self.bandstop(fp2, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        a1 = self.bandstop(a1, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        a2 = self.bandstop(a2, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_ma = self.bandstop(fp1_ma, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_ma = self.bandstop(fp2_ma, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_fp2 = self.bandstop(fp1_fp2, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        chin = self.bandstop(chin, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        m1_m2 = self.bandstop(m1_m2, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_m1 = self.bandstop(fp1_m1, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_m1 = self.bandstop(fp2_m1, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_m2 = self.bandstop(fp1_m2, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_m2 = self.bandstop(fp2_m2, sampling_freq, ffp3, ffs3, gpass2, gstop2)

        """
        --------------------------
        processing for artifact detection
        --------------------------
        """
        fp1_m2_detection = self.highpass(fp1_m2_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp2_m1_detection = self.highpass(fp2_m1_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp1_ma_detection = self.highpass(fp1_ma_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp2_ma_detection = self.highpass(fp2_ma_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp1_detection = self.highpass(fp1_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp2_detection = self.highpass(fp2_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        a1_detection = self.highpass(a1_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        a2_detection = self.highpass(a2_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        fp1_fp2_detection = self.highpass(fp1_fp2_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        m1_m2_detection = self.highpass(m1_m2_detection, sampling_freq, fp_high, fs_high, gpass_high, gstop_high)
        chin_detection = self.bandpass(chin_detection, sampling_freq, fp_chin, fs_chin, gpass_chin, gstop_chin)

        fp1_m2_detection = self.bandstop(fp1_m2_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_m1_detection = self.bandstop(fp2_m1_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_ma_detection = self.bandstop(fp1_ma_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_ma_detection = self.bandstop(fp2_ma_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_detection = self.bandstop(fp1_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp2_detection = self.bandstop(fp2_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        a1_detection = self.bandstop(a1_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        a2_detection = self.bandstop(a2_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        fp1_fp2_detection = self.bandstop(fp1_fp2_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        m1_m2_detection = self.bandstop(m1_m2_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)
        chin_detection = self.bandstop(chin_detection, sampling_freq, ffp2, ffs2, gpass2, gstop2)

        fp1_m2_detection = self.bandstop(fp1_m2_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_m1_detection = self.bandstop(fp2_m1_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_ma_detection = self.bandstop(fp1_ma_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_ma_detection = self.bandstop(fp2_ma_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_detection = self.bandstop(fp1_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp2_detection = self.bandstop(fp2_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        a1_detection = self.bandstop(a1_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        a2_detection = self.bandstop(a2_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        fp1_fp2_detection = self.bandstop(fp1_fp2_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        m1_m2_detection = self.bandstop(m1_m2_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)
        chin_detection = self.bandstop(chin_detection, sampling_freq, ffp3, ffs3, gpass2, gstop2)

        num_of_epoch = int(len(fp1_ma) / (30 * sampling_freq))

        # eeg
        fp1_ma = fp1_ma[: num_of_epoch * 30 * sampling_freq]
        fp2_ma = fp2_ma[: num_of_epoch * 30 * sampling_freq]
        m1_m2 = m1_m2[: num_of_epoch * 30 * sampling_freq]
        chin = chin[: num_of_epoch * 30 * sampling_freq]

        fp1_fp2 = fp1_fp2[: num_of_epoch * 30 * sampling_freq]
        fp1_m1 = fp1_m1[: num_of_epoch * 30 * sampling_freq]
        fp2_m1 = fp2_m1[: num_of_epoch * 30 * sampling_freq]
        fp1_m2 = fp1_m2[: num_of_epoch * 30 * sampling_freq]
        fp2_m2 = fp2_m2[: num_of_epoch * 30 * sampling_freq]
        fp1 = fp1[: num_of_epoch * 30 * sampling_freq]
        fp2 = fp2[: num_of_epoch * 30 * sampling_freq]
        a1 = a1[: num_of_epoch * 30 * sampling_freq]
        a2 = a2[: num_of_epoch * 30 * sampling_freq]

        # artifact detection
        fp1_m2_detection = fp1_m2_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp2_m1_detection = fp2_m1_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp1_ma_detection = fp1_ma_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp2_ma_detection = fp2_ma_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp1_detection = fp1_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp2_detection = fp2_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        a1_detection = a1_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        a2_detection = a2_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        fp1_fp2_detection = fp1_fp2_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        m1_m2_detection = m1_m2_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)
        chin_detection = chin_detection[: num_of_epoch * 30 * sampling_freq].reshape(-1, 30 * sampling_freq)

        # create pandas
        columns_list = []
        signal_list = ["fp1_m2", "fp2_m1", "fp1_ma", "fp2_ma", "fp1", "fp2", "a1", "a2", "fp1_fp2", "m1_m2", "chin"]
        for sig_name in signal_list:
            for i in range(30 * sampling_freq):
                columns_list.append(f"{sig_name}_{i}")

        artifact_detection_ndarray = np.concatenate(
            [
                fp1_m2_detection,
                fp2_m1_detection,
                fp1_ma_detection,
                fp2_ma_detection,
                fp1_detection,
                fp2_detection,
                a1_detection,
                a2_detection,
                fp1_fp2_detection,
                m1_m2_detection,
                chin_detection
            ],
            axis=1
        )

        # get result id information
        file_name = os.path.basename(edf_filenames)
        index_name = file_name.split('_')
        index_base_name = f'{index_name[0]}_{index_name[1]}_epoch'

        for i in range(artifact_detection_ndarray.shape[0]):
            # create eeg data
            data_name = f'{index_base_name}{i + 1:04d}'
            if not i:
                artifact_detection_df = pd.DataFrame(
                    artifact_detection_ndarray[i].reshape(-1, 66000), index=[data_name], columns=columns_list
                )
            else:
                tmp_df = pd.DataFrame(
                    artifact_detection_ndarray[i].reshape(-1, 66000), index=[data_name], columns=columns_list
                )
                artifact_detection_df = pd.concat(
                    [
                        artifact_detection_df,
                        tmp_df
                    ]
                )

        # normalize
        sts = StandardScaler()
        normalization_artifact_detection = sts.fit_transform(artifact_detection_df)

        artifact_detection_index_list = artifact_detection_df.index
        artifact_detection_columns_list = artifact_detection_df.columns

        normalization_artifact_detection_df = pd.DataFrame(
            normalization_artifact_detection,
            index=artifact_detection_index_list,
            columns=artifact_detection_columns_list
        )

        # divide each signal
        normalization_fp1_m2_detection = normalization_artifact_detection_df.loc[:, 'fp1_m2_0':'fp1_m2_5999']
        normalization_fp2_m1_detection = normalization_artifact_detection_df.loc[:, 'fp2_m1_0':'fp2_m1_5999']
        normalization_fp1_ma_detection = normalization_artifact_detection_df.loc[:, 'fp1_ma_0':'fp1_ma_5999']
        normalization_fp2_ma_detection = normalization_artifact_detection_df.loc[:, 'fp2_ma_0':'fp2_ma_5999']
        normalization_fp1_detection = normalization_artifact_detection_df.loc[:, 'fp1_0':'fp1_5999']
        normalization_fp2_detection = normalization_artifact_detection_df.loc[:, 'fp2_0':'fp2_5999']
        normalization_a1_detection = normalization_artifact_detection_df.loc[:, 'a1_0':'a1_5999']
        normalization_a2_detection = normalization_artifact_detection_df.loc[:, 'a2_0':'a2_5999']
        normalization_fp1_fp2_detection = normalization_artifact_detection_df.loc[:, 'fp1_fp2_0':'fp1_fp2_5999']
        normalization_m1_m2_detection = normalization_artifact_detection_df.loc[:, 'm1_m2_0':'m1_m2_5999']
        normalization_chin_detection = normalization_artifact_detection_df.loc[:, 'chin_0':'chin_5999']

        # prepare ICA + kurtosis
        normal_fp1_m2_epoch_detection = normalization_fp1_m2_detection.values
        normal_fp2_m1_epoch_detection = normalization_fp2_m1_detection.values
        normal_fp1_ma_epoch_detection = normalization_fp1_ma_detection.values
        normal_fp2_ma_epoch_detection = normalization_fp2_ma_detection.values
        normal_fp1_epoch_detection = normalization_fp1_detection.values
        normal_fp2_epoch_detection = normalization_fp2_detection.values
        normal_a1_epoch_detection = normalization_a1_detection.values
        normal_a2_epoch_detection = normalization_a2_detection.values
        normal_fp1_fp2_epoch_detection = normalization_fp1_fp2_detection.values
        normal_m1_m2_epoch_detection = normalization_m1_m2_detection.values
        normal_chin_epoch_detection = normalization_chin_detection.values

        """
        --------------------------------------------
        ICA + kurtosis analysis + EMD + Averaged FFT
        --------------------------------------------
        """
        for i in range(num_of_epoch):
            # read eeg data
            tmp_fp1_m2 = normal_fp1_m2_epoch_detection[i, :].reshape(1, -1)
            tmp_fp2_m1 = normal_fp2_m1_epoch_detection[i, :].reshape(1, -1)
            tmp_fp1_ma = normal_fp1_ma_epoch_detection[i, :].reshape(1, -1)
            tmp_fp2_ma = normal_fp2_ma_epoch_detection[i, :].reshape(1, -1)
            tmp_fp1 = normal_fp1_epoch_detection[i, :].reshape(1, -1)
            tmp_fp2 = normal_fp2_epoch_detection[i, :].reshape(1, -1)
            tmp_a1 = normal_a1_epoch_detection[i, :].reshape(1, -1)
            tmp_a2 = normal_a2_epoch_detection[i, :].reshape(1, -1)
            tmp_fp1_fp2 = normal_fp1_fp2_epoch_detection[i, :].reshape(1, -1)
            tmp_m1_m2 = normal_m1_m2_epoch_detection[i, :].reshape(1, -1)
            tmp_chin = normal_chin_epoch_detection[i, :].reshape(1, -1)

            # ica
            integrated_ica = np.vstack(
                [
                    tmp_fp1_m2,
                    tmp_fp2_m1,
                    tmp_fp1_ma,
                    tmp_fp2_ma,
                    tmp_fp1,
                    tmp_fp2,
                    tmp_a1,
                    tmp_a2,
                    tmp_fp1_fp2,
                    tmp_m1_m2,
                    tmp_chin
                ]
            ).T

            # ica parameters
            n_components = 4

            max_iter = 400
            tol = 0.001

            kurtosis_threshold = 3

            ica = FastICA(
                n_components=n_components, whiten="arbitrary-variance", max_iter=max_iter, tol=tol, random_state=42
            )

            ica_results = ica.fit_transform(integrated_ica)
            ica_results = ica_results.T

            # Artifact identification
            kurts = kurtosis(ica_results, axis=0)

            # A component with large kurtosis is identified as an artifact.
            art_idx = np.where(kurts > kurtosis_threshold)[0]

            # Artifact component removal and reconstruction
            ica_results[art_idx, :] = 0
            chs_denoised = ica.inverse_transform(ica_results.T)
            chs_denoised = chs_denoised.T

            # Extracting Artifact Candidates
            fp1_m2_artifact = tmp_fp1_m2 - chs_denoised[0]
            fp2_m1_artifact = tmp_fp2_m1 - chs_denoised[1]
            fp1_ma_artifact = tmp_fp1_ma - chs_denoised[2]
            fp2_ma_artifact = tmp_fp2_ma - chs_denoised[3]
            fp1_artifact = tmp_fp1 - chs_denoised[4]
            fp2_artifact = tmp_fp2 - chs_denoised[5]
            a1_artifact = tmp_a1 - chs_denoised[6]
            a2_artifact = tmp_a2 - chs_denoised[7]
            fp1_fp2_artifact = tmp_fp1_fp2 - chs_denoised[8]
            m1_m2_artifact = tmp_m1_m2 - chs_denoised[9]
            chin_artifact = tmp_chin - chs_denoised[10]

            # confirm data
            t = np.linspace(0, 30, 6000).reshape(1, -1)

            """
            ------------------------------------
            Handling candidate artifacts
            ------------------------------------
            """
            if graphic_enable:
                # display result
                plt.rcParams['font.size'] = 14
                plt.rcParams['font.family'] = 'Times New Roman'

                fig = plt.figure(figsize=(40, 30))
                ax0 = fig.add_subplot(8, 1, 1)
                ax1 = fig.add_subplot(8, 1, 2)
                ax2 = fig.add_subplot(8, 1, 3)
                ax3 = fig.add_subplot(8, 1, 4)
                ax4 = fig.add_subplot(8, 1, 5)
                ax5 = fig.add_subplot(8, 1, 6)
                ax6 = fig.add_subplot(8, 1, 7)
                ax7 = fig.add_subplot(8, 1, 8)

                for j in range(8):
                    if not j:
                        array_max = np.max(tmp_fp1_m2)
                        array_min = np.min(tmp_fp1_m2)

                        ax0.plot(t.T, tmp_fp1_m2.T, color="blue", label="fp1_m2 raw")
                        ax0.plot(t.T, fp1_m2_artifact.T, color='red', label="fp1_m2 artifact")
                        ax0.set_ylim(array_min, array_max)
                        ax0.legend(loc="upper right")
                    elif j == 1:
                        array_max = np.max(tmp_fp2_m1)
                        array_min = np.min(tmp_fp2_m1)

                        ax1.plot(t.T, tmp_fp2_m1.T, color="blue", label="fp2_m1 raw")
                        ax1.plot(t.T, fp2_m1_artifact.T, color="red", label="fp2_m1 artifact")
                        ax1.set_ylim(array_min, array_max)
                        ax1.legend(loc="upper right")
                    elif j == 2:
                        array_max = np.max(tmp_fp1_ma)
                        array_min = np.min(tmp_fp1_ma)

                        ax2.plot(t.T, tmp_fp1_ma.T, color="blue", label="fp1_ma raw")
                        ax2.plot(t.T, fp1_ma_artifact.T, color="red", label="fp1_ma artifact")
                        ax2.set_ylim(array_min, array_max)
                        ax2.legend(loc="upper right")
                    elif j == 3:
                        array_max = np.max(tmp_fp2_ma)
                        array_min = np.min(tmp_fp2_ma)

                        ax3.plot(t.T, tmp_fp2_ma.T, color="blue", label="fp2_ma raw")
                        ax3.plot(t.T, fp2_ma_artifact.T, color="red", label="fp2_ma artifact")
                        ax3.set_ylim(array_min, array_max)
                        ax3.legend(loc="upper right")
                    elif j == 4:
                        array_max = np.max(tmp_fp1)
                        array_min = np.min(tmp_fp1)

                        ax4.plot(t.T, tmp_fp1.T, color="blue", label="fp1 raw")
                        ax4.plot(t.T, fp1_artifact.T, color="red", label="fp1 artifact")
                        ax4.set_ylim(array_min, array_max)
                        ax4.legend(loc="upper right")
                    elif j == 5:
                        array_max = np.max(tmp_fp2)
                        array_min = np.min(tmp_fp2)

                        ax5.plot(t.T, tmp_fp2.T, color="blue", label="fp2 raw")
                        ax5.plot(t.T, fp2_artifact.T, color="red", label="fp2 artifact")
                        ax5.set_ylim(array_min, array_max)
                        ax5.legend(loc="upper right")
                    elif j == 6:
                        array_max = np.max(tmp_a1)
                        array_min = np.min(tmp_a1)

                        ax6.plot(t.T, tmp_a1.T, color="blue", label="a1 raw")
                        ax6.plot(t.T, a1_artifact.T, color="red", label="a1 artifact")
                        ax6.set_ylim(array_min, array_max)
                        ax6.legend(loc="upper right")
                    elif j == 7:
                        array_max = np.max(tmp_a2)
                        array_min = np.min(tmp_a2)

                        ax7.plot(t.T, tmp_a2.T, color="blue", label="a2 raw")
                        ax7.plot(t.T, a2_artifact.T, color="red", label="a2 artifact")
                        ax7.set_ylim(array_min, array_max)
                        ax7.legend(loc="upper right")

                plt.title(f'EEG_EPOCH_{i + 1}')
                fig.tight_layout()
                fig_path = os.path.join(
                    os.getcwd(), 'output_file', 'image', f'{index_base_name}{i + 1:04d}_eeg_with_artifact.png'
                )
                plt.savefig(fig_path)
                plt.close()

                del fig
                gc.collect()

                # display result
                plt.rcParams['font.size'] = 14
                plt.rcParams['font.family'] = 'Times New Roman'

                fig = plt.figure(figsize=(30, 25))
                ax0 = fig.add_subplot(3, 1, 1)
                ax1 = fig.add_subplot(3, 1, 2)
                ax2 = fig.add_subplot(3, 1, 3)

                for j in range(3):
                    if not j:
                        array_max = np.max(tmp_fp1_fp2)
                        array_min = np.min(tmp_fp1_fp2)

                        ax0.plot(t.T, tmp_fp1_fp2.T, color="blue", label="fp1_fp2 raw")
                        ax0.plot(t.T, fp1_fp2_artifact.T, color='red', label="fp1_fp2 artifact")
                        ax0.set_ylim(array_min, array_max)
                        ax0.legend(loc="upper right")
                    elif j == 1:
                        array_max = np.max(tmp_m1_m2)
                        array_min = np.min(tmp_m1_m2)

                        ax1.plot(t.T, tmp_m1_m2.T, color="blue", label="m1_m2 raw")
                        ax1.plot(t.T, m1_m2_artifact.T, color="red", label="m1_m2 artifact")
                        ax1.set_ylim(array_min, array_max)
                        ax1.legend(loc="upper right")
                    elif j == 2:
                        array_max = np.max(tmp_chin)
                        array_min = np.min(tmp_chin)

                        ax2.plot(t.T, tmp_chin.T, color="blue", label="chin raw")
                        ax2.plot(t.T, chin_artifact.T, color="red", label="chin artifact")
                        ax2.set_ylim(array_min, array_max)
                        ax2.legend(loc="upper right")

                plt.title(f'EEG_EPOCH_{i + 1}')
                fig.tight_layout()
                fig_path = os.path.join(
                    os.getcwd(), 'output_file', 'image', f'{index_base_name}{i + 1:04d}_eog_with_artifact.png'
                )
                plt.savefig(fig_path)
                plt.close()

                del fig
                gc.collect()

            """
            ---------------------------------
            emd analyzer
            --------------------------------
            """
            # fp1_m2
            fp1_m2_artifact_imf = emd.sift.sift(
                fp1_m2_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp1_m2_artifact_imf = np.hstack(fp1_m2_artifact_imf.T)

            # fp2_m1
            fp2_m1_artifact_imf = emd.sift.sift(
                fp2_m1_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp2_m1_artifact_imf = np.hstack(fp2_m1_artifact_imf.T)

            # fp1_ma
            fp1_ma_artifact_imf = emd.sift.sift(
                fp1_ma_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp1_ma_artifact_imf = np.hstack(fp1_ma_artifact_imf.T)

            # fp2_ma
            fp2_ma_artifact_imf = emd.sift.sift(
                fp2_ma_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp2_ma_artifact_imf = np.hstack(fp2_ma_artifact_imf.T)

            # fp1
            fp1_artifact_imf = emd.sift.sift(
                fp1_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp1_artifact_imf = np.hstack(fp1_artifact_imf.T)

            # fp2
            fp2_artifact_imf = emd.sift.sift(
                fp2_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp2_artifact_imf = np.hstack(fp2_artifact_imf.T)

            # a1
            a1_artifact_imf = emd.sift.sift(
                a1_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            a1_artifact_imf = np.hstack(a1_artifact_imf.T)

            # a2
            a2_artifact_imf = emd.sift.sift(
                a2_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            a2_artifact_imf = np.hstack(a2_artifact_imf.T)

            # fp1_fp2
            fp1_fp2_artifact_imf = emd.sift.sift(
                fp1_fp2_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            fp1_fp2_artifact_imf = np.hstack(fp1_fp2_artifact_imf.T)

            # m1_m2
            m1_m2_artifact_imf = emd.sift.sift(
                m1_m2_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            m1_m2_artifact_imf = np.hstack(m1_m2_artifact_imf.T)

            # chin
            chin_artifact_imf = emd.sift.sift(
                chin_artifact.reshape(-1, 1), max_imfs=7, imf_opts={'sd_thresh': 0.2}
            )
            chin_artifact_imf = np.hstack(chin_artifact_imf.T)

            artifact_imf = np.concatenate(
                [
                    fp1_m2_artifact_imf,
                    fp2_m1_artifact_imf,
                    fp1_ma_artifact_imf,
                    fp2_ma_artifact_imf,
                    fp1_artifact_imf,
                    fp2_artifact_imf,
                    a1_artifact_imf,
                    a2_artifact_imf,
                    fp1_fp2_artifact_imf,
                    m1_m2_artifact_imf,
                    chin_artifact_imf
                ]
            )

            # Prepare pandas (index & columns)
            epoch_index = str(artifact_detection_index_list[i])

            columns_time_list = []
            signal_list = [
                'fp1_m2', 'fp2_m1', 'fp1_ma', 'fp2_ma', 'fp1', 'fp2', 'a1', 'a2', 'fp1_fp2', 'm1_m2', 'chin'
            ]
            imf_list = [
                'IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7'
            ]
            for signal in signal_list:
                for imf in imf_list:
                    for j in range(int(30 * sampling_freq)):
                        columns_time_list.append(f'{signal}_{imf}_{j}')

            # Prepare pandas data
            artifact_data_time = pd.DataFrame(
                data=artifact_imf.reshape(1, -1), index=[epoch_index], columns=columns_time_list
            )

            del fp1_m2_artifact_imf
            del fp2_m1_artifact_imf
            del fp1_ma_artifact_imf
            del fp2_ma_artifact_imf
            del fp1_artifact_imf
            del fp2_artifact_imf
            del a1_artifact_imf
            del a2_artifact_imf
            del fp1_fp2_artifact_imf
            del m1_m2_artifact_imf
            del chin_artifact_imf

            """
            -----------------------------------------
            Transform the time domain to the frequency domain.
            -----------------------------------------
            """
            # parameters of averaged FFT
            samplerate = 200
            Fs = 800  # Frame size (4sec * 200Hz = 800)
            overlap = 90  # overlap ratio
            # Creating a frame time axis for graph drawing
            t1 = np.linspace(0, 30, 6000)
            t2 = np.arange(0, Fs) / samplerate

            columns_freq_list = []
            for signal in signal_list:
                for imf in imf_list:
                    for j in range(Fs):
                        columns_freq_list.append(f'{signal}_{imf}_{j}')

            """
            ---------------------------------
            fp1_m2 signal
            ---------------------------------
            """
            fp1_m2_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp1_m2',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp2_m1 signal
            ---------------------------------
            """
            fp2_m1_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp2_m1',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp1_ma signal
            ---------------------------------
            """
            fp1_ma_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp1_ma',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp1_ma signal
            ---------------------------------
            """
            fp2_ma_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp2_ma',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp1 signal
            ---------------------------------
            """
            fp1_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp1',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp2 signal
            ---------------------------------
            """
            fp2_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp2',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            a1 signal
            ---------------------------------
            """
            a1_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='a1',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            a2 signal
            ---------------------------------
            """
            a2_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='a2',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            fp1_fp2 signal
            ---------------------------------
            """
            fp1_fp2_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='fp1_fp2',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            m1_m2 signal
            ---------------------------------
            """
            m1_m2_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='m1_m2',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            """
            ---------------------------------
            chin signal
            ---------------------------------
            """
            chin_artifact_data_freq = transform_from_time_to_frequency_domain(
                training_data=artifact_data_time,
                epoch=epoch_index,
                signal_name='chin',
                t1=t1,
                samplerate=samplerate,
                Fs=Fs,
                overlap=overlap,
                t2=t2,
                graphic_enable=graphic_enable
            )

            artifact_imf_freq = np.concatenate(
                [
                    # fp1_m2
                    fp1_m2_artifact_data_freq.imf1_ndarray,
                    fp1_m2_artifact_data_freq.imf2_ndarray,
                    fp1_m2_artifact_data_freq.imf3_ndarray,
                    fp1_m2_artifact_data_freq.imf4_ndarray,
                    fp1_m2_artifact_data_freq.imf5_ndarray,
                    fp1_m2_artifact_data_freq.imf6_ndarray,
                    fp1_m2_artifact_data_freq.imf7_ndarray,
                    # fp2_m1
                    fp2_m1_artifact_data_freq.imf1_ndarray,
                    fp2_m1_artifact_data_freq.imf2_ndarray,
                    fp2_m1_artifact_data_freq.imf3_ndarray,
                    fp2_m1_artifact_data_freq.imf4_ndarray,
                    fp2_m1_artifact_data_freq.imf5_ndarray,
                    fp2_m1_artifact_data_freq.imf6_ndarray,
                    fp2_m1_artifact_data_freq.imf7_ndarray,
                    # fp1_ma
                    fp1_ma_artifact_data_freq.imf1_ndarray,
                    fp1_ma_artifact_data_freq.imf2_ndarray,
                    fp1_ma_artifact_data_freq.imf3_ndarray,
                    fp1_ma_artifact_data_freq.imf4_ndarray,
                    fp1_ma_artifact_data_freq.imf5_ndarray,
                    fp1_ma_artifact_data_freq.imf6_ndarray,
                    fp1_ma_artifact_data_freq.imf7_ndarray,
                    # fp2_ma
                    fp2_ma_artifact_data_freq.imf1_ndarray,
                    fp2_ma_artifact_data_freq.imf2_ndarray,
                    fp2_ma_artifact_data_freq.imf3_ndarray,
                    fp2_ma_artifact_data_freq.imf4_ndarray,
                    fp2_ma_artifact_data_freq.imf5_ndarray,
                    fp2_ma_artifact_data_freq.imf6_ndarray,
                    fp2_ma_artifact_data_freq.imf7_ndarray,
                    # fp1
                    fp1_artifact_data_freq.imf1_ndarray,
                    fp1_artifact_data_freq.imf2_ndarray,
                    fp1_artifact_data_freq.imf3_ndarray,
                    fp1_artifact_data_freq.imf4_ndarray,
                    fp1_artifact_data_freq.imf5_ndarray,
                    fp1_artifact_data_freq.imf6_ndarray,
                    fp1_artifact_data_freq.imf7_ndarray,
                    # fp2
                    fp2_artifact_data_freq.imf1_ndarray,
                    fp2_artifact_data_freq.imf2_ndarray,
                    fp2_artifact_data_freq.imf3_ndarray,
                    fp2_artifact_data_freq.imf4_ndarray,
                    fp2_artifact_data_freq.imf5_ndarray,
                    fp2_artifact_data_freq.imf6_ndarray,
                    fp2_artifact_data_freq.imf7_ndarray,
                    # a1
                    a1_artifact_data_freq.imf1_ndarray,
                    a1_artifact_data_freq.imf2_ndarray,
                    a1_artifact_data_freq.imf3_ndarray,
                    a1_artifact_data_freq.imf4_ndarray,
                    a1_artifact_data_freq.imf5_ndarray,
                    a1_artifact_data_freq.imf6_ndarray,
                    a1_artifact_data_freq.imf7_ndarray,
                    # a2
                    a2_artifact_data_freq.imf1_ndarray,
                    a2_artifact_data_freq.imf2_ndarray,
                    a2_artifact_data_freq.imf3_ndarray,
                    a2_artifact_data_freq.imf4_ndarray,
                    a2_artifact_data_freq.imf5_ndarray,
                    a2_artifact_data_freq.imf6_ndarray,
                    a2_artifact_data_freq.imf7_ndarray,
                    # fp1_fp2
                    fp1_fp2_artifact_data_freq.imf1_ndarray,
                    fp1_fp2_artifact_data_freq.imf2_ndarray,
                    fp1_fp2_artifact_data_freq.imf3_ndarray,
                    fp1_fp2_artifact_data_freq.imf4_ndarray,
                    fp1_fp2_artifact_data_freq.imf5_ndarray,
                    fp1_fp2_artifact_data_freq.imf6_ndarray,
                    fp1_fp2_artifact_data_freq.imf7_ndarray,
                    # m1_m2
                    m1_m2_artifact_data_freq.imf1_ndarray,
                    m1_m2_artifact_data_freq.imf2_ndarray,
                    m1_m2_artifact_data_freq.imf3_ndarray,
                    m1_m2_artifact_data_freq.imf4_ndarray,
                    m1_m2_artifact_data_freq.imf5_ndarray,
                    m1_m2_artifact_data_freq.imf6_ndarray,
                    m1_m2_artifact_data_freq.imf7_ndarray,
                    # chin
                    chin_artifact_data_freq.imf1_ndarray,
                    chin_artifact_data_freq.imf2_ndarray,
                    chin_artifact_data_freq.imf3_ndarray,
                    chin_artifact_data_freq.imf4_ndarray,
                    chin_artifact_data_freq.imf5_ndarray,
                    chin_artifact_data_freq.imf6_ndarray,
                    chin_artifact_data_freq.imf7_ndarray,
                ]
            )

            if not i:
                # Prepare empty frequency
                artifact_data_freq = pd.DataFrame(
                    data=artifact_imf_freq.reshape(1, -1), index=[epoch_index], columns=columns_freq_list
                )
            else:
                tmp_freq = pd.DataFrame(
                    data=artifact_imf_freq.reshape(1, -1), index=[epoch_index], columns=columns_freq_list
                )

                artifact_data_freq = pd.concat(
                    [
                        artifact_data_freq,
                        tmp_freq
                    ]
                )
        """
        ----------------------------------------------
        Judgment process for the presence or absence of artfact components
        ----------------------------------------------
        """
        # read model
        random_forest_model_path = os.path.join(os.getcwd(), 'assets', 'model', 'model_random_forest_best.pkl')
        light_gbm_model_path = os.path.join(os.getcwd(), 'assets', 'model', 'model_light_gbm__best.pkl')

        random_forest_model = joblib.load(random_forest_model_path)
        light_gbm_model = joblib.load(light_gbm_model_path)

        random_results = random_forest_model.predict(artifact_data_freq)
        light_gbm_model = light_gbm_model.predict(artifact_data_freq)

        # Outputs the index that matches the value in the array.
        matching_epoch = []
        for index_epoch in range(random_results.shape[0]):
            if random_results[index_epoch] == 1 or light_gbm_model[index_epoch] == 1:
                matching_epoch.append(index_epoch + 1)

        # prepare index for pandas
        index_list = []
        for tmp_index in matching_epoch:
            index_list.append(artifact_detection_index_list[tmp_index - 1])

        # output artifact Artifact candidate
        index_list_str = ", ".join(index_list)
        store_path = os.path.join(os.getcwd(), 'output_file', 'results',
                                  f'{index_name[0]}_{index_name[1]}_artifact_candidate.csv')
        with open(store_path, "w") as f:
            f.write(index_list_str)

        columns_list = []
        signal_list = [
            'fp1',
            'fp2',
            'a1',
            'a2',
            'fp1_ma',
            'fp2_ma',
            'fp1_fp2',
            'm1_m2',
            'fp1_m1',
            'fp2_m1',
            'fp1_m2',
            'fp2_m2',
        ]

        for signal_name in signal_list:
            for j in range(30 * sampling_freq):
                columns_list.append(f'{signal_name}_{j}')

        # Change the EEG and EOG data to data with artifact components removed.
        target_artifact_df = pd.DataFrame(columns=columns_list)

        correction_artifact_df = pd.DataFrame(columns=columns_list)

        for j, target_index in enumerate(matching_epoch):
            subject_name = index_list[j]
            # print(f'{subject_name}_processing!')
            # target eeg & eog data
            original_fp1 = fp1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp2 = fp2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_a1 = a1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_a2 = a2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp1_ma = fp1_ma[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp2_ma = fp2_ma[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp1_fp2 = fp1_fp2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_m1_m2 = m1_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp1_m1 = fp1_m1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp2_m1 = fp2_m1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp1_m2 = fp1_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]
            original_fp2_m2 = fp2_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq]

            target_fp1 = copy.copy(original_fp1)
            target_fp2 = copy.copy(original_fp2)
            target_a1 = copy.copy(original_a1)
            target_a2 = copy.copy(original_a2)
            target_fp1_ma = copy.copy(original_fp1_ma)
            target_fp2_ma = copy.copy(original_fp2_ma)
            target_fp1_fp2 = copy.copy(original_fp1_fp2)
            target_m1_m2 = copy.copy(original_m1_m2)
            target_fp1_m1 = copy.copy(original_fp1_m1)
            target_fp2_m1 = copy.copy(original_fp2_m1)
            target_fp1_m2 = copy.copy(original_fp1_m2)
            target_fp2_m2 = copy.copy(original_fp2_m2)
            target_fp1 = target_fp1.reshape(1, -1)
            target_fp2 = target_fp2.reshape(1, -1)
            target_a1 = target_a1.reshape(1, -1)
            target_a2 = target_a2.reshape(1, -1)
            target_fp1_ma = target_fp1_ma.reshape(1, -1)
            target_fp2_ma = target_fp2_ma.reshape(1, -1)
            target_fp1_fp2 = target_fp1_fp2.reshape(1, -1)
            target_m1_m2 = target_m1_m2.reshape(1, -1)
            target_fp1_m1 = target_fp1_m1.reshape(1, -1)
            target_fp2_m1 = target_fp2_m1.reshape(1, -1)
            target_fp1_m2 = target_fp1_m2.reshape(1, -1)
            target_fp2_m2 = target_fp2_m2.reshape(1, -1)

            target_data = np.concatenate(
                [
                    target_fp1,
                    target_fp2,
                    target_a1,
                    target_a2,
                    target_fp1_ma,
                    target_fp2_ma,
                    target_fp1_fp2,
                    target_m1_m2,
                    target_fp1_m1,
                    target_fp2_m1,
                    target_fp1_m2,
                    target_fp2_m2,
                ],
                axis=1
            )

            # create data include artifact by pandas
            if target_artifact_df.empty:
                target_artifact_df = pd.DataFrame(data=target_data, index=[subject_name], columns=columns_list)
            else:
                tmp_df = pd.DataFrame(data=target_data, index=[subject_name], columns=columns_list)
                target_artifact_df = pd.concat(
                    [
                        target_artifact_df,
                        tmp_df
                    ]
                )

            # create correction data
            correction_ica = np.vstack(
                [
                    target_fp1,
                    target_fp2,
                    target_a1,
                    target_a2,
                    target_fp1_ma,
                    target_fp2_ma,
                    target_fp1_fp2,
                    target_m1_m2,
                    target_fp1_m1,
                    target_fp2_m1,
                    target_fp1_m2,
                    target_fp2_m2,
                ]
            ).T

            # ica parameters
            n_components = 4

            max_iter = 400
            tol = 0.001

            kurtosis_threshold = 3

            ica = FastICA(
                n_components=n_components, whiten="arbitrary-variance", max_iter=max_iter, tol=tol, random_state=42
            )

            ica_results = ica.fit_transform(correction_ica)
            ica_results = ica_results.T
            
            """
             Find the correlation coefficient
            between the original signal and the ICA analysis signal.
            """
            # ica_results[0, :]
            original_ica_correlation_fp1 = np.corrcoef(ica_results[0, :], original_fp1)[0, 1]
            original_ica_correlation_fp2 = np.corrcoef(ica_results[0, :], original_fp2)[0, 1]
            original_ica_correlation_a1 = np.corrcoef(ica_results[0, :], original_a1)[0, 1]
            original_ica_correlation_a2 = np.corrcoef(ica_results[0, :], original_a2)[0, 1]
            
            original_ica_correlation_0 = np.abs(np.max(
                [original_ica_correlation_fp1, original_ica_correlation_fp2, original_ica_correlation_a1,
                 original_ica_correlation_a2])
            )
            
            if original_ica_correlation_0 < 0.2:
                # each 3sec
                art_idx = []
                for i in range(ica_results[0, :].shape[0]):
                    if 0 <= i < 300:
                        if i - 300:
                            start_point = 0
                        else:
                            start_point = i - 299
                        end_point = i + 300
                    elif 300 <= i < 5700:
                        start_point = i - 300
                        end_point = i + 300
                    else:
                        start_point = i - 300
                        if i >= 5700:
                            end_point = 6000
                        else:
                            end_point = i + 300
                    
                    # calculate kurts
                    kurts = kurtosis(ica_results[0, start_point:end_point], axis=0)
                    if kurts > kurtosis_threshold:
                        art_idx.append(i)
                        
                ica_results[0, art_idx] = 0
                print("remove ica_results[0, :]!!!!")
            
            # ica_results[1, :]
            original_ica_correlation_fp1 = np.corrcoef(ica_results[1, :], original_fp1)[0, 1]
            original_ica_correlation_fp2 = np.corrcoef(ica_results[1, :], original_fp2)[0, 1]
            original_ica_correlation_a1 = np.corrcoef(ica_results[1, :], original_a1)[0, 1]
            original_ica_correlation_a2 = np.corrcoef(ica_results[1, :], original_a2)[0, 1]
            
            original_ica_correlation_1 = np.abs(np.max(
                [original_ica_correlation_fp1, original_ica_correlation_fp2, original_ica_correlation_a1,
                 original_ica_correlation_a2])
            )
            
            if original_ica_correlation_1 < 0.2:
                # each 3sec
                art_idx = []
                for i in range(ica_results[1, :].shape[0]):
                    if 0 <= i < 300:
                        if i - 300:
                            start_point = 0
                        else:
                            start_point = i - 299
                        end_point = i + 300
                    elif 300 <= i < 5700:
                        start_point = i - 300
                        end_point = i + 300
                    else:
                        start_point = i - 300
                        if i >= 5700:
                            end_point = 6000
                        else:
                            end_point = i + 300
                    
                    # calculate kurts
                    kurts = kurtosis(ica_results[1, start_point:end_point], axis=0)
                    if kurts > kurtosis_threshold:
                        art_idx.append(i)
                
                ica_results[1, art_idx] = 0
                print("remove ica_results[1, :]!!!!")
            
            # ica_results[2, :]
            original_ica_correlation_fp1 = np.corrcoef(ica_results[2, :], original_fp1)[0, 1]
            original_ica_correlation_fp2 = np.corrcoef(ica_results[2, :], original_fp2)[0, 1]
            original_ica_correlation_a1 = np.corrcoef(ica_results[2, :], original_a1)[0, 1]
            original_ica_correlation_a2 = np.corrcoef(ica_results[2, :], original_a2)[0, 1]
            
            original_ica_correlation_2 = np.abs(np.max(
                [original_ica_correlation_fp1, original_ica_correlation_fp2, original_ica_correlation_a1,
                 original_ica_correlation_a2])
            )
            
            if original_ica_correlation_2 < 0.2:
                # each 3sec
                art_idx = []
                for i in range(ica_results[2, :].shape[0]):
                    if 0 <= i < 300:
                        if i - 300:
                            start_point = 0
                        else:
                            start_point = i - 299
                        end_point = i + 300
                    elif 300 <= i < 5700:
                        start_point = i - 300
                        end_point = i + 300
                    else:
                        start_point = i - 300
                        if i >= 5700:
                            end_point = 6000
                        else:
                            end_point = i + 300
                    
                    # calculate kurts
                    kurts = kurtosis(ica_results[2, start_point:end_point], axis=0)
                    if kurts > kurtosis_threshold:
                        art_idx.append(i)
                
                ica_results[2, art_idx] = 0
                print("remove ica_results[2, :]!!!!")
            
            # ica_results[3, :]
            original_ica_correlation_fp1 = np.corrcoef(ica_results[3, :], original_fp1)[0, 1]
            original_ica_correlation_fp2 = np.corrcoef(ica_results[3, :], original_fp2)[0, 1]
            original_ica_correlation_a1 = np.corrcoef(ica_results[3, :], original_a1)[0, 1]
            original_ica_correlation_a2 = np.corrcoef(ica_results[3, :], original_a2)[0, 1]
            
            original_ica_correlation_3 = np.abs(np.max(
                [original_ica_correlation_fp1, original_ica_correlation_fp2, original_ica_correlation_a1,
                 original_ica_correlation_a2])
            )
            
            if original_ica_correlation_3 < 0.2:
                # each 3sec
                art_idx = []
                for i in range(ica_results[3, :].shape[0]):
                    if 0 <= i < 300:
                        if i - 300:
                            start_point = 0
                        else:
                            start_point = i - 299
                        end_point = i + 300
                    elif 300 <= i < 5700:
                        start_point = i - 300
                        end_point = i + 300
                    else:
                        start_point = i - 300
                        if i >= 5700:
                            end_point = 6000
                        else:
                            end_point = i + 300
                    
                    # calculate kurts
                    kurts = kurtosis(ica_results[3, start_point:end_point], axis=0)
                    if kurts > kurtosis_threshold:
                        art_idx.append(i)
                
                ica_results[3, art_idx] = 0
                print("remove ica_results[3, :]!!!!")
            
            chs_denoised = ica.inverse_transform(ica_results.T)
            chs_denoised = chs_denoised.T

            fp1_correction = chs_denoised[0].reshape(1, -1)
            fp2_correction = chs_denoised[1].reshape(1, -1)
            a1_correction = chs_denoised[2].reshape(1, -1)
            a2_correction = chs_denoised[3].reshape(1, -1)
            fp1_ma_correction = chs_denoised[4].reshape(1, -1)
            fp2_ma_correction = chs_denoised[5].reshape(1, -1)
            fp1_fp2_correction = chs_denoised[6].reshape(1, -1)
            m1_m2_correction = chs_denoised[7].reshape(1, -1)
            fp1_m1_correction = chs_denoised[8].reshape(1, -1)
            fp2_m1_correction = chs_denoised[9].reshape(1, -1)
            fp1_m2_correction = chs_denoised[10].reshape(1, -1)
            fp2_m2_correction = chs_denoised[11].reshape(1, -1)

            correction_data = np.concatenate(
                [
                    fp1_correction,
                    fp2_correction,
                    a1_correction,
                    a2_correction,
                    fp1_ma_correction,
                    fp2_ma_correction,
                    fp1_fp2_correction,
                    m1_m2_correction,
                    fp1_m1_correction,
                    fp2_m1_correction,
                    fp1_m2_correction,
                    fp2_m2_correction,
                ],
                axis=1
            )


            # create data include artifact by pandas
            if correction_artifact_df.empty:
                correction_artifact_df = pd.DataFrame(data=correction_data, index=[subject_name], columns=columns_list)
            else:
                tmp_df = pd.DataFrame(data=correction_data, index=[subject_name], columns=columns_list)
                correction_artifact_df = pd.concat(
                    [
                        correction_artifact_df,
                        tmp_df
                    ]
                )

            fp1_correction = fp1_correction.reshape(-1, 1)
            fp2_correction = fp2_correction.reshape(-1, 1)
            a1_correction = a1_correction.reshape(-1, 1)
            a2_correction = a2_correction.reshape(-1, 1)
            fp1_ma_correction = fp1_ma_correction.reshape(-1, 1)
            fp2_ma_correction = fp2_ma_correction.reshape(-1, 1)
            fp1_fp2_correction = fp1_fp2_correction.reshape(-1, 1)
            m1_m2_correction = m1_m2_correction.reshape(-1, 1)
            fp1_m1_correction = fp1_m1_correction.reshape(-1, 1)
            fp2_m1_correction = fp2_m1_correction.reshape(-1, 1)
            fp1_m2_correction = fp1_m2_correction.reshape(-1, 1)
            fp2_m2_correction = fp2_m2_correction.reshape(-1, 1)

            #if graphic_enable:
            # difference data
            plt.rcParams['font.size'] = 14
            plt.rcParams['font.family'] = 'Times New Roman'

            fig = plt.figure(figsize=(40, 30))
            ax0 = fig.add_subplot(6, 1, 1)
            ax1 = fig.add_subplot(6, 1, 2)
            ax2 = fig.add_subplot(6, 1, 3)
            ax3 = fig.add_subplot(6, 1, 4)
            ax4 = fig.add_subplot(6, 1, 5)
            ax5 = fig.add_subplot(6, 1, 6)

            for j in range(6):
                if not j:
                    array_max = np.max(original_fp1)
                    array_min = np.min(original_fp1)

                    fp1_correction_flat = fp1_correction.flatten()
                    dif_fp1 = original_fp1 - fp1_correction_flat
                    ax0.plot(t1, original_fp1, linewidth=0.1, color="blue", alpha=1)
                    ax0.plot(t1, fp1_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax0.plot(t1, dif_fp1, linewidth=0.5, color="black", alpha=1)
                    ax0.set_ylim(array_min, array_max)
                    ax0.set_title("fp1 dif")
                elif j == 1:
                    array_max = np.max(original_fp2)
                    array_min = np.min(original_fp2)

                    fp2_correction_flat = fp2_correction.flatten()
                    dif_fp2 = original_fp2 - fp2_correction_flat
                    ax1.plot(t1, original_fp2, linewidth=0.1, color="blue", alpha=1)
                    ax1.plot(t1, fp2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax1.plot(t1, dif_fp2, linewidth=0.5, color="black", alpha=1)
                    ax1.set_ylim(array_min, array_max)
                    ax1.set_title("fp2 dif")
                elif j == 2:
                    array_max = np.max(original_a1)
                    array_min = np.min(original_a1)

                    a1_correction_flat = a1_correction.flatten()
                    dif_a1 = original_a1 - a1_correction_flat
                    ax2.plot(t1, original_a1, linewidth=0.1, color="blue", alpha=1)
                    ax2.plot(t1, a1_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax2.plot(t1, dif_a1, linewidth=0.5, color="black", alpha=1)
                    ax2.set_ylim(array_min, array_max)
                    ax2.set_title("a1 dif")
                elif j == 3:
                    array_max = np.max(original_a2)
                    array_min = np.min(original_a2)

                    a2_correction_flat = a2_correction.flatten()
                    dif_a2 = original_a2 - a2_correction_flat
                    ax3.plot(t1, original_a2, linewidth=0.1, color="blue", alpha=1)
                    ax3.plot(t1, a2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax3.plot(t1, dif_a2, linewidth=0.5, color="black", alpha=1)
                    ax3.set_ylim(array_min, array_max)
                    ax3.set_title("a2 dif")
                elif j == 4:
                    array_max = np.max(original_fp1_ma)
                    array_min = np.min(original_fp1_ma)

                    fp1_ma_correction_flat = fp1_ma_correction.flatten()
                    dif_fp1_ma = original_fp1_ma - fp1_ma_correction_flat
                    ax4.plot(t1, original_fp1_ma, linewidth=0.1, color="blue", alpha=1)
                    ax4.plot(t1, fp1_ma_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax4.plot(t1, dif_fp1_ma, linewidth=0.5, color="black", alpha=1)
                    ax4.set_ylim(array_min, array_max)
                    ax4.set_title("fp1_ma dif")
                elif j == 5:
                    array_max = np.max(original_fp2_ma)
                    array_min = np.min(original_fp2_ma)

                    fp2_ma_correction_flat = fp2_ma_correction.flatten()
                    dif_fp2_ma = original_fp2_ma - fp2_ma_correction_flat
                    ax5.plot(t1, original_fp2_ma, linewidth=0.1, color="blue", alpha=1)
                    ax5.plot(t1, fp2_ma_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax5.plot(t1, dif_fp2_ma, linewidth=0.5, color="black", alpha=1)
                    ax5.set_ylim(array_min, array_max)
                    ax5.set_title("fp2_ma dif")

            plt.tight_layout()

            if graphic_enable:
                fig_path = os.path.join(
                    os.getcwd(), 'output_file', 'image', 'correction', f'{subject_name}_dif_raw_correction_1.png'
                )
                plt.savefig(fig_path)
            plt.close()

            del fig
            del dif_fp1
            del dif_fp2
            del dif_a1
            del dif_a2
            del dif_fp1_ma
            del dif_fp2_ma

            gc.collect()

            # difference data
            plt.rcParams['font.size'] = 14
            plt.rcParams['font.family'] = 'Times New Roman'

            fig = plt.figure(figsize=(40, 30))
            ax0 = fig.add_subplot(6, 1, 1)
            ax1 = fig.add_subplot(6, 1, 2)
            ax2 = fig.add_subplot(6, 1, 3)
            ax3 = fig.add_subplot(6, 1, 4)
            ax4 = fig.add_subplot(6, 1, 5)
            ax5 = fig.add_subplot(6, 1, 6)

            for j in range(6):
                if not j:
                    array_max = np.max(original_fp1_fp2)
                    array_min = np.min(original_fp1_fp2)

                    fp1_fp2_correction_flat = fp1_fp2_correction.flatten()
                    dif_fp1_fp2 = original_fp1_fp2 - fp1_fp2_correction_flat
                    ax0.plot(t1, original_fp1_fp2, linewidth=0.1, color="blue", alpha=1)
                    ax0.plot(t1, fp1_fp2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax0.plot(t1, dif_fp1_fp2, linewidth=0.5, color="black", alpha=1)
                    ax0.set_ylim(array_min, array_max)
                    ax0.set_title("fp1_fp2 dif")
                elif j == 1:
                    array_max = np.max(original_m1_m2)
                    array_min = np.min(original_m1_m2)

                    m1_m2_correction_flat = m1_m2_correction.flatten()
                    dif_m1_m2 = original_m1_m2 - m1_m2_correction_flat
                    ax1.plot(t1, original_m1_m2, linewidth=0.1, color="blue", alpha=1)
                    ax1.plot(t1, m1_m2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax1.plot(t1, dif_m1_m2, linewidth=0.5, color="black", alpha=1)
                    ax1.set_ylim(array_min, array_max)
                    ax1.set_title("m1_m2 dif")
                elif j == 2:
                    array_max = np.max(original_fp1_m1)
                    array_min = np.min(original_fp1_m1)

                    fp1_m1_correction_flat = fp1_m1_correction.flatten()
                    dif_fp1_m1 = original_fp1_m1 - fp1_m1_correction_flat
                    ax2.plot(t1, original_fp1_m1, linewidth=0.1, color="blue", alpha=1)
                    ax2.plot(t1, fp1_m1_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax2.plot(t1, dif_fp1_m1, linewidth=0.5, color="black", alpha=1)
                    ax2.set_ylim(array_min, array_max)
                    ax2.set_title("fp1_m1 dif")
                elif j == 3:
                    array_max = np.max(original_fp2_m1)
                    array_min = np.min(original_fp2_m1)

                    fp2_m1_correction_flat = fp2_m1_correction.flatten()
                    dif_fp2_m1 = original_fp2_m1 - fp2_m1_correction_flat
                    ax3.plot(t1, original_fp2_m1, linewidth=0.1, color="blue", alpha=1)
                    ax3.plot(t1, fp2_m1_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax3.plot(t1, dif_fp2_m1, linewidth=0.5, color="black", alpha=1)
                    ax3.set_ylim(array_min, array_max)
                    ax3.set_title("fp2_m1 dif")
                elif j == 4:
                    array_max = np.max(original_fp1_m2)
                    array_min = np.min(original_fp1_m2)

                    fp1_m2_correction_flat = fp1_m2_correction.flatten()
                    dif_fp1_m2 = original_fp1_m2 - fp1_m2_correction_flat
                    ax4.plot(t1, original_fp1_m2, linewidth=0.1, color="blue", alpha=1)
                    ax4.plot(t1, fp1_m2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax4.plot(t1, dif_fp1_m2, linewidth=0.5, color="black", alpha=1)
                    ax4.set_ylim(array_min, array_max)
                    ax4.set_title("fp1_m2 dif")
                elif j == 5:
                    array_max = np.max(original_fp2_m2)
                    array_min = np.min(original_fp2_m2)

                    fp2_m2_correction_flat = fp2_m2_correction.flatten()
                    dif_fp2_m2 = original_fp2_m2 - fp2_m2_correction_flat
                    ax5.plot(t1, original_fp2_m2, linewidth=0.1, color="blue", alpha=1)
                    ax5.plot(t1, fp2_m2_correction_flat, linewidth=0.1, color="red", alpha=1)
                    ax5.plot(t1, dif_fp2_m2, linewidth=0.5, color="black", alpha=1)
                    ax5.set_ylim(array_min, array_max)
                    ax5.set_title("fp2_m2 dif")

            plt.tight_layout()

            if graphic_enable:
                fig_path = os.path.join(
                    os.getcwd(), 'output_file', 'image', 'correction', f'{subject_name}_dif_raw_correction_2.png'
                )
                plt.savefig(fig_path)
            plt.close()

            del fig
            del dif_fp1_fp2
            del dif_m1_m2
            del dif_fp1_m1
            del dif_fp2_m1
            del dif_fp1_m2
            del dif_fp2_m2

            gc.collect()

            # Swapping data
            fp1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = fp1_correction_flat
            fp2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = fp2_correction_flat
            a1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = a1_correction_flat
            a2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = a2_correction_flat
            fp1_ma[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp1_ma_correction_flat
            )
            fp2_ma[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp2_ma_correction_flat
            )
            fp1_fp2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp1_fp2_correction_flat
            )
            m1_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                m1_m2_correction_flat
            )
            fp1_m1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp1_m1_correction_flat
            )
            fp2_m1[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp2_m1_correction_flat
            )
            fp1_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp1_m2_correction_flat
            )
            fp2_m2[(target_index - 1) * 30 * sampling_freq: target_index * 30 * sampling_freq] = (
                fp2_m2_correction_flat
            )

            # memory release
            del target_fp1
            del target_fp2
            del target_a1
            del target_a2
            del target_fp1_ma
            del target_fp2_ma
            del target_fp1_fp2
            del target_m1_m2
            del target_fp1_m1
            del target_fp2_m1
            del target_fp1_m2
            del target_fp2_m2

            del fp1_correction
            del fp2_correction
            del a1_correction
            del a2_correction
            del fp1_ma_correction
            del fp2_ma_correction
            del fp1_fp2_correction
            del m1_m2_correction
            del fp1_m1_correction
            del fp2_m1_correction
            del fp1_m2_correction
            del fp2_m2_correction

            del fp1_correction_flat
            del fp2_correction_flat
            del a1_correction_flat
            del a2_correction_flat
            del fp1_ma_correction_flat
            del fp2_ma_correction_flat
            del fp1_fp2_correction_flat
            del m1_m2_correction_flat
            del fp1_m1_correction_flat
            del fp2_m1_correction_flat
            del fp1_m2_correction_flat
            del fp2_m2_correction_flat

        # store calculation results
        if store_enable:
            raw_data_path = os.path.join(os.getcwd(), 'output_file', f'{index_name[0]}_{index_name[1]}_target_raw.csv')
            target_artifact_df.to_csv(raw_data_path)

            correction_data_path = os.path.join(
                os.getcwd(), 'output_file', f'{index_name[0]}_{index_name[1]}_target_correction.csv'
            )
            correction_artifact_df.to_csv(correction_data_path)

            # memory release
            del target_artifact_df
            del correction_artifact_df

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
            chin=chin,
            light_off=lightsoff,
            light_on=lightson,
            elect_info=elect_info,
            duration=duration,
        )

    @staticmethod
    def bandpass(x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2
        wp = f_p / f_n
        ws = (
                f_s / f_n
        )

        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)

        b, a = signal.butter(nn, wn, "band")
        y = signal.filtfilt(b, a, x)
        return y

    @staticmethod
    def highpass(x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2
        wp = f_p / f_n
        ws = f_s / f_n
        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)
        b, a = signal.butter(nn, wn, btype='high')
        y = signal.filtfilt(b, a, x)
        return y

    @staticmethod
    def bandstop(x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2
        wp = f_p / f_n
        ws = (
                f_s / f_n
        )

        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)

        b, a = signal.butter(nn, wn, "bandstop")
        y = signal.filtfilt(b, a, x)
        return y
