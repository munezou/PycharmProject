# これはサンプルの Python スクリプトです。

# Shift+F10 を押して実行するか、ご自身のコードに置き換えてください。
# Shift を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。
from numpy import ndarray


def main():
    import tracemalloc
    
    def format_bytes(size):
        power = 2 ** 10  # 2**10 = 1024
        nn = 0
        power_labels = ['B', 'MB', 'GB', 'TB']
        while size > power and nn <= len(power_labels):
            size /= power
            nn += 1
        return 'current used memory: {:.3f} {}'.format(size, power_labels[nn])
    
    def log_memory():
        snapshot = tracemalloc.take_snapshot()
        size = sum([stat.size for stat in snapshot.statistics('filename')])
        print(format_bytes(size))
    
    tracemalloc.start()
    print('-- start --')
    log_memory()
    
    # python library
    import glob
    import time  # to measure analysis time.
    import random
    import xml.etree.ElementTree as et
    import pyedflib
    import csv
    import math
    import os
    import train_func as func_ph21
    import traceback
    
    
    # numpy library
    random.seed(0)
    import numpy as np
    np.random.seed(6391)
    
    # tensorflow library
    from tensorflow.keras import backend as k
    from tensorflow.python.keras.layers import Layer
    from tensorflow.keras.layers import ELU
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate
    from tensorflow.keras.layers import Conv1D, Input, MaxPool1D
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from tensorflow.keras.layers import BatchNormalization
    
    # sklearn library
    from sklearn.metrics import confusion_matrix, cohen_kappa_score
    
    # definition of custum objects
    class LayerNormalization(Layer):
        try:
            def __init__(self, **kwargs):
                super(LayerNormalization, self).__init__(**kwargs)
                self.epsilon = 1e-6
            
            def build(self, input_shape):
                self.built = False
            
            def call(self, x):
                mean = k.mean(x, axis=-1, keepdims=True)
                std = k.std(x, axis=-1, keepdims=True)
                norm = (x - mean) * (1 / (std + self.epsilon))
                return norm
            
            def compute_output_shape(self, input_shape):
                return input_shape
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
    
    
    import numpy as np
    from scipy import signal
    samplerate = 200  # 300(Ph1.5)
    
    # バターワースフィルタ（バンドパス）
    def bandpass(x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2  # ナイキスト周波数
        wp = f_p / f_n  # ナイキスト周波数で通過域端周波数を正規化
        ws = f_s / f_n  # ナイキスト周波数で阻止域端周波数を正規化
        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)  # オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(nn, wn, "band")  # フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
        return y
    
    def bandstop(x, sample_rate, f_p, f_s, g_pass, g_stop):
        f_n = sample_rate / 2  # ナイキスト周波数
        wp = f_p / f_n  # ナイキスト周波数で通過域端周波数を正規化
        ws = f_s / f_n  # ナイキスト周波数で阻止域端周波数を正規化
        nn, wn = signal.buttord(wp, ws, g_pass, g_stop)  # オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(nn, wn, "bandstop")  # フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
        return y
    
    print('-- import all --')
    log_memory()
    
    bai = 1
    bai2 = 8
    fb = 1
    
    def stage_model_main3ch():
        try:
            # MAIN PART
            eeg_input_main = Input(shape=(int(math.floor(6000)), 3), dtype='float32', name='eeg')
            eeg = Conv1D(
                32 * fb,
                50 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg = ELU()(eeg)
            eeg = Conv1D(
                32 * fb,
                10 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs2'
            )(eeg)
            
            eeg = ELU()(eeg)  # exponential linear unit as activation function
            eeg = Conv1D(
                16 * fb,
                10 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs3'
            )(eeg)
            
            eeg = ELU()(eeg)
            eeg = MaxPool1D(pool_size=10)(eeg)  # Max pooling layer
            eeg = LayerNormalization(eeg)
            
            eeg2 = Conv1D(
                32 * fb,
                50 * bai2,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fl1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg2 = ELU()(eeg2)
            eeg2 = Conv1D(32 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl2')(eeg2)
            eeg2 = ELU()(eeg2)  # exponential linear unit as activation function
            eeg2 = Conv1D(16 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl3')(eeg2)
            eeg2 = ELU()(eeg2)
            eeg2 = MaxPool1D(pool_size=10)(eeg2)  # Max pooling layer
            eeg2 = LayerNormalization(eeg2)
            
            features = concatenate([eeg, eeg2])  # ([eeg,eog,emg,cam])
            features = Conv1D(5, 10, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fcam')(features)  # 50
            x_main = GlobalAveragePooling1D()(features)
            # x_main = GlobalMaxPooling1D()(features)
            main_output = Activation('softmax', name='stages')(x_main)
            
            return Model(inputs=[eeg_input_main], outputs=[main_output])
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass

    def stage_model_1ch():
        try:
            # MAIN PART
            eeg_input_main = Input(shape=(int(math.floor(6000)), 1), dtype='float32', name='eeg')
            eeg = Conv1D(
                32 * fb,
                50 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg = ELU(eeg)
            eeg = Conv1D(32 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs2')(eeg)
            eeg = ELU(eeg)  # exponential linear unit as activation function
            eeg = Conv1D(16 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs3')(eeg)
            eeg = ELU(eeg)
            eeg = MaxPool1D(pool_size=10)(eeg)  # Max pooling layer
            eeg = LayerNormalization(eeg)
            
            eeg2 = Conv1D(
                32 * fb,
                50 * bai2,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fl1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg2 = ELU(eeg2)
            eeg2 = Conv1D(32 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl2')(eeg2)
            eeg2 = ELU(eeg2)  # exponential linear unit as activation function
            eeg2 = Conv1D(16 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl3')(eeg2)
            eeg2 = ELU(eeg2)
            eeg2 = MaxPool1D(pool_size=10)(eeg2)  # Max pooling layer
            eeg2 = LayerNormalization(eeg2)
            
            features = concatenate([eeg, eeg2])  # ([eeg,eog,emg,cam])
            features = Conv1D(5, 10, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fcam')(features)  # 50
            x_main = GlobalAveragePooling1D()(features)
            # x_main = GlobalMaxPooling1D()(features)
            main_output = Activation('softmax', name='stages')(x_main)
            
            return Model(inputs=[eeg_input_main], outputs=[main_output])
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass

    def stage_model_2ch():
        try:
            # MAIN PART
            eeg_input_main = Input(shape=(int(math.floor(6000)), 2), dtype='float32', name='eeg')
            eeg = Conv1D(
                32 * fb,
                50 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg = ELU(eeg)
            eeg = Conv1D(
                32 * fb,
                10 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same', name='fs2'
            )(eeg)
            
            eeg = ELU(eeg)  # exponential linear unit as activation function
            eeg = Conv1D(16 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs3')(eeg)
            eeg = ELU(eeg)
            eeg = MaxPool1D(pool_size=10)(eeg)  # Max pooling layer
            eeg = LayerNormalization(eeg)
            
            eeg2 = Conv1D(
                32 * fb,
                50 * bai2,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fl1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg2 = ELU(eeg2)
            eeg2 = Conv1D(32 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl2')(eeg2)
            eeg2 = ELU(eeg2)  # exponential linear unit as activation function
            eeg2 = Conv1D(16 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl3')(eeg2)
            eeg2 = ELU(eeg2)
            eeg2 = MaxPool1D(pool_size=10)(eeg2)  # Max pooling layer
            eeg2 = LayerNormalization(eeg2)
            
            features = concatenate([eeg, eeg2])  # ([eeg,eog,emg,cam])
            features = Conv1D(5, 10, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fcam')(features)  # 50
            x_main = GlobalAveragePooling1D()(features)
            # x_main = GlobalMaxPooling1D()(features)
            main_output = Activation('softmax', name='stages')(x_main)
            
            return Model(inputs=[eeg_input_main], outputs=[main_output])
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
    
    def stage_model_4ch():
        try:
            # MAIN PART
            eeg_input_main = Input(shape=(int(math.floor(6000)), 4), dtype='float32', name='eeg')
            eeg = Conv1D(
                32 * fb, 50 * bai,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fs1')(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg = ELU(eeg)
            eeg = Conv1D(32 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs2')(eeg)
            eeg = ELU(eeg)  # exponential linear unit as activation function
            eeg = Conv1D(16 * fb, 10 * bai, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fs3')(eeg)
            eeg = ELU(eeg)
            eeg = MaxPool1D(pool_size=10)(eeg)  # Max pooling layer
            eeg = LayerNormalization(eeg)
            
            eeg2 = Conv1D(
                32 * fb, 50 * bai2,
                kernel_regularizer=regularizers.l2(0.001),
                padding='same',
                name='fl1'
            )(eeg_input_main)  # convolutional layer 1 50  filter96
            
            eeg2 = ELU(eeg2)
            eeg2 = Conv1D(32 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl2')(eeg2)
            eeg2 = ELU(eeg2)  # exponential linear unit as activation function
            eeg2 = Conv1D(16 * fb, 10 * bai2, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fl3')(eeg2)
            eeg2 = ELU(eeg2)
            eeg2 = MaxPool1D(pool_size=10)(eeg2)  # Max pooling layer
            eeg2 = LayerNormalization(eeg2)
            
            features = concatenate([eeg, eeg2])  # ([eeg,eog,emg,cam])
            features = Conv1D(5, 10, kernel_regularizer=regularizers.l2(0.001), padding='same', name='fcam')(features)  # 50
            x_main = GlobalAveragePooling1D()(features)
            # x_main = GlobalMaxPooling1D()(features)
            main_output = Activation('softmax', name='stages')(x_main)
            
            return Model(inputs=[eeg_input_main], outputs=[main_output])
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
    
    
    dim = int(16 * 2 * 2 + 5 * 2)
    
    
    def stages_mean_max_norm():
        # MAIN PART (int(math.floor(1)),5)
        preds_input_main = Input(shape=(int(math.floor(11)), dim), dtype='float32', name='pred')
        normed_in = BatchNormalization()(preds_input_main)
        # x_main=Flatten()(preds_input_main)
        x_main = Flatten()(normed_in)
        x_main = Dense(5)(x_main)
        main_output = Activation('softmax', name='stages')(x_main)
        
        return Model(inputs=[preds_input_main], outputs=[main_output])
    
    """
    ---------------------------------------------------------------------------------------
                                          evaluation
    ---------------------------------------------------------------------------------------
    """
    
    print('-- before --')
    log_memory()

    print(f"----< import all >---")
    func_ph21.log_memory()

    print(f"----< before >----")
    func_ph21.log_memory()
    
    t0 = time.time()

    # setting current directory
    project_directory = os.getcwd()

    # confirm whether required files exit or not
    edf_filter = os.path.join(project_directory, "Ph21data", "*.edf")
    rml_filter = os.path.join(project_directory, "Ph21data", "*.rml")
    csv_filter = os.path.join(project_directory, "Ph21data", "*.csv")
    csv_ct_filter = os.path.join(project_directory, "Ph21data", "*-ct.csv")

    # create file list
    edf_list = glob.glob(edf_filter)
    edf_list.sort()
    edf_list_size = len(edf_list)
    print(f"a number of edf_list: {edf_list_size}")
    
    rml_list = glob.glob(rml_filter)
    rml_list.sort()
    rml_list_size = len(rml_list)
    print(f"a number of rml_list: {rml_list_size}")
    
    csv_list = glob.glob(csv_filter)
    csv_list.sort()

    csv_ct_list = glob.glob(csv_ct_filter)
    csv_ct_list.sort()
    csv_ct_list_size = len(csv_ct_list)
    print(f"a number of csv_ct_list: {csv_ct_list_size}")
    
    for csv_ct_file in csv_ct_list:
        csv_list.remove(csv_ct_file)
    csv_list_size = len(csv_list)
    print(f"a number of csv_list: {csv_list_size}\n")

    # extend file name
    if not len(edf_list):
        # Output message.
        print(f"edf files don't exist!\n")
        # end
        exit()
    else:
        # Delete duplicate files
        edf_hash = func_ph21.delete_duplicate_edf(edf_list)
        edf_list = list(edf_hash.values())

    if not len(rml_list):
        # output message.
        print(f"rml files don't exist!\n")
        # end
        exit()
    else:
        rml_hash = func_ph21.delete_duplicate_edf(rml_list)
        rml_list = list(rml_hash.values())

    if not len(csv_list):
        # output message.
        print(f"csv files don't exist!\n")
        # end
        exit()
    else:
        csv_hash = func_ph21.delete_duplicate_edf(csv_list)
        csv_list = list(csv_hash.values())

    if not len(csv_ct_list):
        # output message.
        print(f"csv_ct files don't exist!\n")
        # end
        exit()
    else:
        csv_ct_hash = func_ph21.delete_duplicate_edf(csv_ct_list)
        csv_ct_list = list(csv_ct_hash.values())
    
    fold = rml_list
    fold.sort()

    fold_n = np.zeros(len(fold))

    for n, fn in enumerate(fold):
        fold_n[n] = int(fold[n].split('_')[-2][-9:])

    flag_p = 0
    weakNames = ['0base', '1base', '2xFp1', '3xFp2', '4xM1', '5xM2', '6xFp1M1', '7xFp1M2', '8xFp2M1', '9xFp2M2',
                 '4ch_raw', '4ch']
    tag_n_all = []
    paths = []
    
    ntnum = 0
    set_n = 0
    
    print('set:', set_n)
    fnum = 0
    
    # print(dnum,dpath)
    path = fold
    fpath_ct = csv_ct_list
    
    try:
        dpath = path[set_n]
        tree = et.parse(dpath)
        root = tree.getroot()
        t_dur = root.findall('.//{http://www.respironics.com/PatientStudy.xsd}Duration')
        duration = int(t_dur[0].text)
        n_epo = int(duration / 30)
        
        stage = root.findall(
            './/{http://www.respironics.com/PatientStudy.xsd}UserStaging//{http://www.respironics.com/PatientStudy.xsd}Stage'
        )
    except Exception as ex:
        print(f"error information: {traceback.format_exc()}")
        pass
    
    eeg_tra = None
    ss_tra = None
    ss_tra2 = None
    whotra = None
    setall = None
    setall2 =None
    n_tra = None

    predall = np.array([])
    predall2 = np.array([])
    
    if len(stage) > 0:
        st_num = np.zeros(len(stage))
        st_epo = np.zeros(len(stage))
        
        for num, st in enumerate(stage):
            # to quantify the stage.
            st_num[num] = int(
                st.attrib['Type'].replace('NotScored', '-1').replace('Wake', '0').replace('NonREM1', '2').replace(
                    'NonREM2', '3').replace('NonREM3', '4').replace('REM', '1'))
            # to stores the transition number of stage.
            st_epo[num] = int(st.attrib['Start']) / 30 + 1
        
        stage_all = np.zeros(n_epo)
        
        for i, _ in enumerate(stage[:-1]):
            stage_all[int(st_epo[i] - 1):int(st_epo[i + 1])] = st_num[i]
        stage_all[int(st_epo[-1] - 1):] = st_num[-1]
        
        event = []
        try:
            with open(fpath_ct[set_n], 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    event.append(row[-1])
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
        
        try:
            stage_all2 = [
                # to quantify the stage.
                s.replace('NS', '-1').replace('WK', '0').replace('N1', '2').replace('N2', '3').replace('N3', '4').replace('REM', '1') for s in event
            ]
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass

        try:
            # stage_all2 = np.array(stage_all2).astype(int)[:n_epo]
            stage_all2 = np.array(stage_all2)
            stage_all2 = stage_all2[stage_all2 != '睡眠ステージ']
            stage_all2 = stage_all2.astype(int)[:n_epo]
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
        
        eves = root.findall('.//{http://www.respironics.com/PatientStudy.xsd}Event')
        
        arou_all = []
        
        for ev in eves:
            if ev.attrib['Type'] == 'Arousal':
                arou_all.append([float(ev.attrib['Start']), float(ev.attrib['Duration'])])
        
        tag_full = np.zeros(int(duration * 2))
        arou_len_all_pre = np.zeros(len(arou_all))
        for numa, tag in enumerate(arou_all):
            tag_full[int(tag[0] * 2):int((tag[0] + tag[1]) * 2)] = 1
            arou_len_all_pre[numa] = tag[1]
        
        edf = pyedflib.EdfReader(dpath.replace(".rml", ".edf"))
        
        [ch0, ch1, ch2, ch3] = [0, 1, 2, 3]
        
        fp1 = edf.readSignal(ch0)
        fp1 = fp1[:n_epo * 30 * 200]
        fp2 = edf.readSignal(ch1)
        fp2 = fp2[:n_epo * 30 * 200]
        a1 = edf.readSignal(ch2)
        a1 = a1[:n_epo * 30 * 200]
        a2 = edf.readSignal(ch3)
        a2 = a2[:n_epo * 30 * 200]
        ma = (a1 + a2) / 2
        fp1_ma = fp1 - ma
        fp2_ma = fp2 - ma
        fp1_m2 = fp1 - a2
        fp2_m2 = fp2 - a2
        fp1_m1 = fp1 - a1
        fp2_m1 = fp2 - a1
        fp1_fp2 = fp1 - fp2
        m1_m2 = a1 - a2
        
        nu1 = edf.readSignal(4)  # Light off
        nu2 = edf.readSignal(5)  # R_A1
        nu3 = edf.readSignal(6)  # R_Fp1
        nu4 = edf.readSignal(7)  # R_Ref
        nu5 = edf.readSignal(8)  # R_Fp2
        nu6 = edf.readSignal(9)  # R_A2
        nu1 = nu1[29::30].reshape(-1, 1)
        nu2 = nu2[29::30].reshape(-1, 1)
        nu3 = nu3[29::30].reshape(-1, 1)
        nu4 = nu4[29::30].reshape(-1, 1)
        nu5 = nu5[29::30].reshape(-1, 1)
        nu6 = nu6[29::30].reshape(-1, 1)
        
        edf.close()
        
        del edf
        
        elect_n = np.concatenate([nu1, nu2, nu3, nu4, nu5, nu6], axis=1)
        elect_n = elect_n.astype(np.float32)

        fp = np.array([0.3, 35])  # k-d
        fs = np.array([0.15, 70])

        gpass = -3  # 3                       #通過域端最大損失[dB]
        gstop = -5.5  # 40                      #阻止域端最小損失[dB]
        
        # fp2 = np.array([49.9, 50.1])  # k-d
        fs2 = np.array([45, 55])
        gpass2 = 3  # 3                       #通過域端最大損失[dB]
        gstop2 = 40  # 40                      #阻止域端最小損失[dB]
        
        try:
            eeg1f = fp1_ma.reshape(n_epo, -1, 1)
            eeg2f = fp2_ma.reshape(n_epo, -1, 1)
            eeg3f = fp1_fp2.reshape(n_epo, -1, 1)
            eeg4f = m1_m2.reshape(n_epo, -1, 1)
            eeg5f = fp1_m1.reshape(n_epo, -1, 1)
            eeg6f = fp2_m1.reshape(n_epo, -1, 1)
            eeg7f = fp1_m2.reshape(n_epo, -1, 1)
            eeg8f = fp2_m2.reshape(n_epo, -1, 1)
            eeg9f = fp1.reshape(n_epo, -1, 1)
            eeg10f = fp2.reshape(n_epo, -1, 1)
            eeg11f = a1.reshape(n_epo, -1, 1)
            eeg12f = a2.reshape(n_epo, -1, 1)
        except Exception as ex:
            print(f"error information: {traceback.format_exc()}")
            pass
        
        eeg_concf = np.concatenate(
            [eeg1f, eeg2f, eeg3f, eeg4f, eeg5f, eeg6f, eeg7f, eeg8f, eeg9f, eeg10f, eeg11f, eeg12f], axis=2)
        eeg_n = eeg_concf.astype(np.float32)
        # StageN=Stage_all[Stage_all>-1]
        
        stage_n = stage_all
        stage_n2 = stage_all2
        
        # st=np.arange(len(StageN))[StageN>-1][0]
        # en=np.arange(len(StageN))[StageN>-1][-1]
        # StageN=StageN[st:en]
        # eegN=eegN[st:en]
        # Tag_allN=tag_full
        if eeg_n.shape[0] > 60:
            paths.append(dpath)
            eeg_tra = eeg_n
            ss_tra = stage_n
            ss_tra2 = stage_n2
            whotra = np.ones(len(stage_n)) * fold_n[fnum]  # subject number
            n_tra = np.ones(len(stage_n)) * ntnum  # tooshi number for val
            electtes = elect_n
            ntnum += 1
            tag_n_all.append(len(arou_all))
    
    print("total")
    print("tes:", eeg_tra.shape, ss_tra.shape, whotra.shape)
    print(np.sum(ss_tra == -1), np.sum(ss_tra == 0), np.sum(ss_tra == 1), np.sum(ss_tra == 2), np.sum(ss_tra == 3),
          np.sum(ss_tra == 4), np.sum(ss_tra > -5))
    print(np.sum(ss_tra2 == -1), np.sum(ss_tra2 == 0), np.sum(ss_tra2 == 1), np.sum(ss_tra2 == 2), np.sum(ss_tra2 == 3),
          np.sum(ss_tra2 == 4), np.sum(ss_tra2 > -5))
    # print("val:",EEGval.shape,SSval.shape,Whoval.shape)
    t1 = time.time()
    print(t1 - t0)
    log_memory()
    
    chs = None
    ss_all = None
    ss_all2 = None

    predkakushin = np.array([])
    predkakushin2 = np.empty([])

    for weak_n in range(10):  # 12
        # weakN=1
        print(set_n, weakNames[weak_n])
        loaded = np.load(f"Ph21_npz//{str(set_n)}_ph21_noise_mean_std_50cut2_{weakNames[weak_n]}.npz")
        mean_e = loaded['mean_E']
        std_e = loaded['std_E']
        # mean_E=np.mean(EEGtra[SStra>-1].reshape(-1,3),axis=0)
        # std_E=np.std(EEGtra[SStra>-1].reshape(-1,3),axis=0)
        print(mean_e)
        print(std_e)
        # np.savez(str(set_n)+'_ph21_noise_mean_std_1base.npz',mean_E=mean_E,std_E=std_E)
        t2 = time.time()
        print(t2 - t1)
        print(set_n, 'mean and std are loaded!')
        
        if weak_n == 0:
            chs = [0, 1, 2]  # [eeg1,eeg2,eeg3]=[Fp1Ma,Fp2Ma,Fp1Fp2]
        elif weak_n == 1:
            chs = [0, 1, 3]  # [eeg1,eeg2,eeg3]=[Fp1Ma,Fp2Ma,M1M2]
        elif weak_n == 4:
            chs = [6, 7, 2]  # [eeg1,eeg2,eeg3]=[Fp1M2,Fp2M2,Fp1Fp2]
        elif weak_n == 5:
            chs = [4, 5, 2]  # [eeg1,eeg2,eeg3]=[Fp1M1,Fp2M1,Fp1Fp2]
        elif weak_n == 2:
            chs = [1, 3]  # [eeg1,eeg2]=[Fp2Ma,M1M2]
        elif weak_n == 3:
            chs = [0, 3]  # [eeg1,eeg2]=[Fp1Ma,M1M2]
        elif weak_n == 6:
            chs = [7]  # eeg1=Fp2M2
        elif weak_n == 7:
            chs = [5]  # eeg1=Fp2M1
        elif weak_n == 8:
            chs = [6]  # eeg1=Fp1M2
        elif weak_n == 9:
            chs = [4]  # eeg1=Fp1M1
        elif weak_n in [-1, 11]:
            chs = [0, 1, 2, 3]  # [eeg1,eeg2,eeg3,eeg4]=[Fp1Ma,Fp2Ma,Fp1Fp2,M1M2]
        elif weak_n in [-2, 10]:
            chs = [8, 9, 10, 11]
        else:
            print("Can't found weakN!")
            exit()  # temp [eeg1,eeg2,eeg3,eeg4]=[Fp1,Fp2,A1,A2]
        
        eeg_tra_p = eeg_tra[:, :, chs]
        eeg_tra_n = (eeg_tra_p - mean_e) / std_e
        
        if weak_n in [0, 1, 4, 5]:
            model = stage_model_main3ch()
        elif weak_n in [2, 3]:
            model = stage_model_2ch()
        elif weak_n in [-1, -2, 10, 11]:
            model = stage_model_4ch()
        else:
            model = stage_model_1ch()
        
        model.load_weights(f"Ph21_h5//{str(set_n)}_ph21_noise_50cut2_{weakNames[weak_n]}.h5")
        
        ss_tra = ss_tra2
        predictions = model.predict([eeg_tra_n])
        pred_y = np.argmax(predictions, axis=1)
        
        if weak_n == 0:  # or weakN==0:
            eeg_tra2 = eeg_tra[:, :, [8, 9, 10, 11]]
            yzx = (eeg_tra2[:, :-1, :] * eeg_tra2[:, 1:, :] < 0)
            maxs = np.max(eeg_tra2, axis=1)
            mins = np.min(eeg_tra2, axis=1)
            stds = np.std(eeg_tra2, axis=1)
            zxs = np.sum(yzx, axis=1)
            parall_p = np.stack([maxs, mins, stds, zxs])
        
        # Pred_kakushin_all[Whotes]=predictions
        cm = confusion_matrix(ss_tra[ss_tra > -1], pred_y[ss_tra > -1])
        kappa = cohen_kappa_score(ss_tra[ss_tra > -1], pred_y[ss_tra > -1])
        print(cm)
        print(np.round(cm / np.sum(cm, axis=1).reshape(5, 1) * 100, 1))
        print(f'acc: {sum(ss_tra[ss_tra > -1] == pred_y[ss_tra > -1]) / len(pred_y[ss_tra > -1])}')
        print(f'kappa: {kappa}')
        print("\n")
        
        eeg_tes_n = eeg_tra_n
        cam_model = Model(inputs=model.input, outputs=model.layers[-3].output)
        cam = cam_model.predict([eeg_tes_n])
        cam_tes = np.concatenate([np.mean(cam, axis=1), np.max(cam, axis=1)], axis=1)
        
        feature_s_model = Model(inputs=model.input, outputs=model.layers[-6].output)
        feature = feature_s_model.predict([eeg_tes_n])
        feature_s_tes = np.concatenate([np.mean(feature, axis=1), np.max(feature, axis=1)], axis=1)
        
        feature_l_model = Model(inputs=model.input, outputs=model.layers[-5].output)
        feature = feature_l_model.predict([eeg_tes_n])
        feature_l_tes = np.concatenate([np.mean(feature, axis=1), np.max(feature, axis=1)], axis=1)
        
        n_tes = n_tra
        ss_tes = ss_tra
        whotes = whotra
        print('Features of ', str(set_n), ' are calculated')
        features_pre_tes = np.concatenate([feature_s_tes, feature_l_tes, cam_tes], axis=1)
        dim = features_pre_tes.shape[1]
        
        predtes = None
        f_ss_tes = None
        f_whotes = None
        f_n_tes = None
        
        for tes_id in np.unique(n_tes):
            predtes0_p = features_pre_tes[n_tes == tes_id]
            predtes_p = np.zeros([predtes0_p.shape[0] - 10, 11, dim])
            for i_epo in range(predtes_p.shape[0]):
                predtes_p[i_epo] = predtes0_p[i_epo:i_epo + 11]
            f_ss_tes_p = ss_tes[n_tes == tes_id][5:-5]
            f_whotes_p = whotes[n_tes == tes_id][5:-5]
            f_n_tes_p = n_tes[n_tes == tes_id][5:-5]
            if tes_id == np.unique(n_tes)[0]:
                predtes = predtes_p
                f_ss_tes = f_ss_tes_p
                f_whotes = f_whotes_p
                f_n_tes = f_n_tes_p
            else:
                predtes = np.concatenate([predtes, predtes_p])
                f_ss_tes = np.concatenate([f_ss_tes, f_ss_tes_p])
                f_whotes = np.concatenate([f_whotes, f_whotes_p])
                f_n_tes = np.concatenate([f_n_tes, f_n_tes_p])
        print('test data ready')
        
        t3 = time.time()
        print(t3 - t2)
        
        model = stages_mean_max_norm()
        model.load_weights('./stage_weights/' + str(set_n) + '_ph21_noise_feature_50cut2_' + weakNames[weak_n] + '.h5')
        
        predictions2 = model.predict([predtes])
        pred_y2 = np.argmax(predictions2, axis=1)
        
        # Predall2[Whotes]=pred_y2
        # Pred_kakushin_all2[Whotes]=predictions2
        cm2 = confusion_matrix(f_ss_tes[f_ss_tes > -1], pred_y2[f_ss_tes > -1])
        kappa2 = cohen_kappa_score(f_ss_tes[f_ss_tes > -1], pred_y2[f_ss_tes > -1])
        print(cm2)
        print(np.round(cm2 / np.sum(cm2, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(f_ss_tes[f_ss_tes > -1] == pred_y2[f_ss_tes > -1]) / len(pred_y2[f_ss_tes > -1]))
        print('kappa:', kappa2)
        print('\n')
        
        electtes = None
        setall = np.empty()
        setall2 = np.empty()
        e_all = np.empty()
        
        if flag_p == 0:
            predall = pred_y
            predkakushin = predictions
            ss_all = ss_tra
            predall2 = pred_y2
            predkakushin2 = predictions2
            ss_all2 = f_ss_tes
            flag_p = 1
            e_all = electtes
        else:
            predall = np.concatenate([predall, pred_y])
            predkakushin = np.concatenate([predkakushin, predictions], axis=0)
            predall2 = np.concatenate([predall2, pred_y2])
            predkakushin2 = np.concatenate([predkakushin2, predictions2], axis=0)
            if weak_n == 0:  # or weakN==0:
                ss_all = np.concatenate([ss_all, ss_tra])
                ss_all2 = np.concatenate([ss_all2, f_ss_tes])
                # Pred_kakushin_all[Whotes]=predictions
                e_all = np.concatenate([e_all, electtes], axis=0)
    
    l_old = 0
    l_old2 = 0
    
    if set_n == 0:
        l_old = len(predall)
        l_old2 = len(predall2)
        print(l_old, l_old2)
        setall = np.zeros(l_old)
        setall2 = np.zeros(l_old2)
    else:
        setall = np.concatenate([setall, set_n * np.ones(len(predall) - l_old)])
        setall2 = np.concatenate([setall2, set_n * np.ones(len(predall2) - l_old2)])
        print(l_old, len(predall), l_old2, len(predall2))
    
    log_memory()
    print('-fin-')
    
    roop = 10  # 12
    set_n = 1
    
    predalls = predall[setall == 0].reshape(roop, -1)
    predalls2 = predall2[setall2 == 0].reshape(roop, -1)
    predalls_kakushin = predkakushin[setall == 0].reshape(roop, -1, 5)
    predalls2_kakushin = predkakushin2[setall2 == 0].reshape(roop, -1, 5)
    
    for i in range(set_n - 1):
        predalls = np.concatenate([predalls, predall[setall == i + 1].reshape(roop, -1)], axis=1)
        predalls2 = np.concatenate([predalls2, predall2[setall2 == i + 1].reshape(roop, -1)], axis=1)
        predalls_kakushin = np.concatenate(
            [
                predalls_kakushin,
                predkakushin[setall == i + 1].reshape(roop, -1, 5)
            ],
            axis=1
        )
        predalls2_kakushin = np.concatenate(
            [predalls2_kakushin, predkakushin2[setall2 == i + 1].reshape(roop, -1, 5)],
            axis=1
        )
        
    print(predalls.shape, ss_all.shape)

    e_all = None
    
    for wN in range(roop):
        # wN=2
        print(wN, weakNames[wN])
        cm_a = confusion_matrix(ss_all[ss_all > -1], predalls[wN][ss_all > -1])
        kappa_a = cohen_kappa_score(ss_all[ss_all > -1], predalls[wN][ss_all > -1])
        print(cm_a)
        print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all[ss_all > -1] == predalls[wN][ss_all > -1]) / len(predalls[wN][ss_all > -1]))
        print('kappa:', kappa_a)
        
        cm_f = confusion_matrix(ss_all2[ss_all2 > -1], predalls2[wN][ss_all2 > -1])
        kappa_f = cohen_kappa_score(ss_all2[ss_all2 > -1], predalls2[wN][ss_all2 > -1])
        print('\n 11epoch')
        print(cm_f)
        print(np.round(cm_f / np.sum(cm_f, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all2[ss_all2 > -1] == predalls2[wN][ss_all2 > -1]) / len(predalls2[wN][ss_all2 > -1]))
        print('kappa:', kappa_f)
        print()
        print()
    
    predalls_ens5 = np.argmax(np.mean(predalls_kakushin[0:6], axis=0), axis=1)
    print('ens5')
    cm_a = confusion_matrix(ss_all[ss_all > -1], predalls_ens5[ss_all > -1])
    kappa_a = cohen_kappa_score(ss_all[ss_all > -1], predalls_ens5[ss_all > -1])
    print(cm_a)
    print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
    print('acc:', sum(ss_all[ss_all > -1] == predalls_ens5[ss_all > -1]) / len(predalls_ens5[ss_all > -1]))
    print('kappa:', kappa_a)
    print()
    
    predalls2_ens5 = np.argmax(np.mean(predalls2_kakushin[0:6], axis=0), axis=1)
    print('ens5')
    cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predalls2_ens5[ss_all2 > -1])
    kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predalls2_ens5[ss_all2 > -1])
    print(cm_a)
    print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
    print('acc:', sum(ss_all2[ss_all2 > -1] == predalls2_ens5[ss_all2 > -1]) / len(predalls2_ens5[ss_all2 > -1]))
    print('kappa:', kappa_a)
    print()
    
    predalls_ens9 = np.argmax(np.mean(predalls_kakushin, axis=0), axis=1)
    print('ens9')
    cm_a = confusion_matrix(ss_all[ss_all > -1], predalls_ens9[ss_all > -1])
    kappa_a = cohen_kappa_score(ss_all[ss_all > -1], predalls_ens9[ss_all > -1])
    print(cm_a)
    print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
    print('acc:', sum(ss_all[ss_all > -1] == predalls_ens9[ss_all > -1]) / len(predalls_ens9[ss_all > -1]))
    print('kappa:', kappa_a)
    print()
    
    predalls2_ens9 = np.argmax(np.mean(predalls2_kakushin, axis=0), axis=1)
    print('ens9')
    cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predalls2_ens9[ss_all2 > -1])
    kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predalls2_ens9[ss_all2 > -1])
    print(cm_a)
    print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
    print('acc:', sum(ss_all2[ss_all2 > -1] == predalls2_ens9[ss_all2 > -1]) / len(predalls2_ens9[ss_all2 > -1]))
    print('kappa:', kappa_a)
    print()
    
    arg_np = np.sum(e_all[(ss_all > -1), 1:6] == 100, axis=0)
    arg_n = np.argmax(np.concatenate([arg_np[0:2], arg_np[3:5]]))
    max_n = np.max(np.concatenate([arg_np[0:2], arg_np[3:5]]))
    
    mo1 = 0
    mo2 = 0
    mo3 = 0
    
    if arg_n == 1:
        [mo1, mo2, mo3] = [1, 5, 6]
    elif arg_n == 2:
        [mo1, mo2, mo3] = [2, 7, 8]
    elif arg_n == 0:
        [mo1, mo2, mo3] = [3, 5, 7]
    elif arg_n == 3:
        [mo1, mo2, mo3] = [4, 6, 8]

    mo1 += 1
    mo2 += 1
    mo3 += 1

    predall2_nokori3 = np.argmax(
        (predalls2_kakushin[mo1][(ss_all2 > -1)] + predalls2_kakushin[mo2][(ss_all2 > -1)] + predalls2_kakushin[mo3][
            (ss_all2 > -1)]) / 3,
        axis=1
    )
    predall2_nokori1 = predalls2[mo1][(ss_all2 > -1)]
    
    ratio = (max_n / np.sum((ss_all > -1)))
    
    print('final output')
    if ratio < 0.1:
        print('no noise: 10 ensemble')
        predalls2_ens9 = np.argmax(np.mean(predalls2_kakushin, axis=0), axis=1)
        cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predalls2_ens9[ss_all2 > -1])
        kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predalls2_ens9[ss_all2 > -1])
        print(cm_a)
        print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all2[ss_all2 > -1] == predalls2_ens9[ss_all2 > -1]) / len(predalls2_ens9[ss_all2 > -1]))
        print('kappa:', kappa_a)
        print()
        
        print('no noise: 6 ensemble')
        predalls2_ens5 = np.argmax(np.mean(predalls2_kakushin[0:6], axis=0), axis=1)
        cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predalls2_ens5[ss_all2 > -1])
        kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predalls2_ens5[ss_all2 > -1])
        print(cm_a)
        print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all2[ss_all2 > -1] == predalls2_ens5[ss_all2 > -1]) / len(predalls2_ens5[ss_all2 > -1]))
        print('kappa:', kappa_a)
    else:
        print('noisy! noisy ch:', arg_n, arg_np)
        print('3ensemble')
        cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predall2_nokori3)
        kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predall2_nokori3)
        print(cm_a)
        print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all2[ss_all2 > -1] == predall2_nokori3) / len(predall2_nokori3))
        print('kappa:', kappa_a)
        print()
        
        print('1model without noisy signal')
        cm_a = confusion_matrix(ss_all2[ss_all2 > -1], predall2_nokori1)
        kappa_a = cohen_kappa_score(ss_all2[ss_all2 > -1], predall2_nokori1)
        print(cm_a)
        print(np.round(cm_a / np.sum(cm_a, axis=1).reshape(5, 1) * 100, 1))
        print('acc:', sum(ss_all2[ss_all2 > -1] == predall2_nokori1) / len(predall2_nokori1))
        print('kappa:', kappa_a)


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    main()

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
