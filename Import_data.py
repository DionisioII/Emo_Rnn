import csv
import sys
import array

import numpy as np

CSV_CLASSIFICATION_VALUE_INDEX = 166

def import_data(file):

    print ("Started reading training dataset")
        # Reads data file. Gets X and Y dataset for analysis
    data_cvs = open(file, 'r')
    data_cvs.readline()
    data_cvsreader = csv.reader(data_cvs)





    # Loop through file and process string
    X_data = []
    Y_data = []
    max_frames_for_videos = 0
    invalid_row_count = 0
    count = 0
    iter=0
    video_name = "null"
    temp_class = "null"
    X_data.append([])


    for row in data_cvsreader:

        if video_name != row[165] and count!= 0:
            
            classification_value = temp_class
            Y_data.append(classification_value)
                

            if classification_value == "hap" :
                
                Y_data[iter] = 0
            elif classification_value == "neu" :
                Y_data[iter] = 1
            elif classification_value == "sad" :
                Y_data[iter] = 2
            elif classification_value == "ang" :
                Y_data[iter] = 3
            else:
                invalid_row_count += 1 
                #Invalid value. do not consider this row data
            iter+=1
            X_data.append([])
            count=0
            values =[]
            for i in range(0,165):
                if row[i] != 'NaN':
                    values.append(float(row[i]))
                else:
                    values.append(0.0)
            X_data[iter].append(np.array(list(values)))

        else:
            values =[]
            for i in range(0,165):
                if row[i] != 'NaN':
                    values.append(float(row[i]))
                else:
                    values.append(0.0)
            X_data[iter].append(np.array(list(values)))
        
        count+=1
        if count >max_frames_for_videos:
            max_frames_for_videos = count        
        video_name= row[165]
        temp_class = row[166]
            

    data_cvs.close()
    Y_data.append(1) #cambia con temp_class
    #print(Y_data)

    Y_np_data = np.array(Y_data)
    data_size = len(Y_data)

    X_np_data = np.array(list(X_data))
    #X_np_data = X_np_data.reshape(-1,29,165)
    #print(len(X_np_data[150][5]))

    return X_np_data, Y_np_data, max_frames_for_videos

