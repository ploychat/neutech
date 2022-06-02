import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, utils, applications, initializers
from tensorflow.keras.applications import imagenet_utils
import tensorflow.keras.backend as K

import numpy as np
import pickle

from tensorflow.keras import layers, models, datasets, preprocessing, utils, applications, initializers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
from numbers_parser import Document
from readnumber import *
from eeg_open import *


def eeg_open(name):
    name = "Data/"+name+"/"+name+"_processedEEG_1000.dat"
    with open(name, "rb") as fp:
        EEG_data_l = pickle.load(fp)
        fp.close()  

    TP9_l = EEG_data_l[:, 0, :]
    AF7_l = EEG_data_l[:, 1, :]
    AF8_l = EEG_data_l[:, 2, :]
    TP10_l= EEG_data_l[:, 3, :]

    # print(EEG_data_l)

    return EEG_data_l

def merge_eeg_phq9(pathname):
        
    path_2_updata_l = pathname

    ## Fetch newest PHQ9
    doc_update_l = Document(path_2_updata_l)
    sheets_update_l = doc_update_l.sheets()
    tables_update_l = sheets_update_l[0].tables()
    rows_update_l = tables_update_l[0].rows(values_only=True)

    PHQ9_list_l = []

    PHQ9_id_firstname_l = {}
    PHQ9_id_lastname_l = {}

    data = []
    group0, group1, group2, group3, group4 = 0, 0, 0, 0, 0 #from mildest to most severe

    line_count = 0
    #calculate PHQ-9 of each participant
    for row in rows_update_l:
        if line_count == 0: # Skip the table header        
            line_count += 1
        else:
            id_l = (f'\t{row[1]}').strip()
            firstname_l = (f'\t{row[2]}').strip()
            lastname_l = (f'\t{row[3]}').strip()

            PHQ9_id_firstname_l[id_l] = firstname_l
            PHQ9_id_lastname_l[id_l] = lastname_l       
        
            phq9_1_l = int(row[16])
            phq9_2_l = int(row[17])
            phq9_3_l = int(row[18])
            phq9_4_l = int(row[19])
            phq9_5_l = int(row[20])
            phq9_6_l = int(row[21])
            phq9_7_l = int(row[22])
            phq9_8_l = int(row[23])
            phq9_9_l = int(row[24])

            PHQ9_group = PHQ9_score_2_group(phq9_1_l + phq9_2_l + phq9_3_l + phq9_4_l + phq9_5_l + phq9_6_l + phq9_7_l + phq9_8_l + phq9_9_l)

            PHQ9_list_l.append(PHQ9_group)
            if PHQ9_group == 0:
                group0 += 1
            elif PHQ9_group == 1:
                group1 += 1  
            elif PHQ9_group == 2:
                group2 += 1 
            elif PHQ9_group == 3:
                group3 += 1 
            else:
                group4 += 1                    
            data.append([id_l, PHQ9_group, eeg_open(id_l)])
            line_count += 1
    return data
    
data = merge_eeg_phq9("round2EEG.numbers")

df = pd.DataFrame(data, columns=('ID', 'Group', 'EEGData'))
for _ in df['EEGData']:
    print(df['ID'])
    print(_.shape)
