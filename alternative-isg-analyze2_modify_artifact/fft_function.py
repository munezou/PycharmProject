import os
import numpy as np
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt


# Overlap processing
def ov(data, samplerate, Fs, overlap):
    # all data length
    Ts = len(data) / samplerate
    
    # frame frequency
    Fc = Fs / samplerate
    
    # Frame shift width when overlapping
    x_ol = Fs * (1 - (overlap/100))
    
    # Number of frames to extract (number of data used for averaging)
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100))))
 
    # Define an empty array to hold the extracted data
    array = []
 
    #Extract data with for loop.
    for i in range(N_ave):
        # Update the cut position for each loop.
        ps = int(x_ol * i)
        
        # Extract the frame size from the cut position ps and add it to the array.
        array.append(data[ps:ps+Fs:1])
    
    # Use the data array and the number of data that are overlap extracted as return values.
    return array, N_ave                 


# Window function processing (Hanning window)
def hanning(data_array, Fs, N_ave):
    # Create a Hanning window
    han = signal.hann(Fs)
    
    # Amplitude Correction Factor
    acf = 1 / (sum(han) / Fs)
 
    # Window function is applied to all overlapped multi-time waveforms.
    for i in range(N_ave):
        # Apply a window function.
        data_array[i] = data_array[i] * han
 
    return data_array, acf


# FFT processing
def fft_ave(data_array,samplerate, Fs, N_ave, acf):
    fft_array = []
    for i in range(N_ave):
        # FFT is added to the array, window function correction value is applied, 
        # and (Fs/2) normalization is performed.
        fft_array.append(acf*np.abs(fftpack.fft(data_array[i])/(Fs/2)))
 
    # Create a frequency axis.
    fft_axis = np.linspace(0, samplerate, Fs)
    
    # Convert type to ndarray.
    fft_array = np.array(fft_array)
    
    # Calculate the average power of all FFT waveforms 
    # and then use it as the amplitude value.
    fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))
    
    return fft_array, fft_mean, fft_axis


# fft plot
def plot_data(emd_index, signal_name, samplerate, N_ave, t, time_array, fft_axis, fft_mean):
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    # Turn the scale inside.
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Add gridlines to the top, bottom, left, and right of the graph.
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(212)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    # Set axis labels.
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Signal')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Signal')
    
    time_array_max = np.abs(np.max(time_array))
    time_array_min = np.abs(np.min(time_array))
    if time_array_max >= time_array_min:
        ax1_yRange = time_array_max
    else:
        ax1_yRange = time_array_min
    
    fft_mean_max = np.abs(np.max(fft_mean))
    fft_mean_min = np.abs(np.min(fft_mean))
    if fft_mean_max >= fft_mean_min:
        ax2_yRange = fft_mean_max
    else:
        ax2_yRange = fft_mean_min

    # Explicit data range and increments.
    ax1.set_xlim(0, 4)
    ax1.set_ylim((-1) * ax1_yRange, ax1_yRange)
    ax2.set_xticks(np.arange(0, samplerate, 50))
    ax2.set_yticks(np.arange(0, 3, 0.5))
    ax2.set_xlim(0,200)
    ax2.set_ylim(0, ax2_yRange)

    # Along with preparing data plots, set labels, line thicknesses, and legends.
    for i in range(N_ave):
        ax1.plot(t, time_array[i], label='signal', lw=1)

    ax2.plot(fft_axis, fft_mean, label='signal', lw=1)

    fig.tight_layout()

    # save graphic
    file_path = os.path.join(
        os.getcwd(), 'output_file', 'image', 'averaged_fft', f'{emd_index}_{signal_name}_artifact.png'
    )
    plt.savefig(file_path)

    # View graphs.
    # plt.show()
    plt.close()
