import numpy as np 
import SimpleITK as sitk 
import pandas as pd 
import os
import matplotlib.pyplot as plt 

os.chdir(r'/home/spl/Hands-on-machine-learning/Codes/100x100x48 niis')
file_list = []

y = pd.read_csv(r'/media/spl/D/MicroCT_data/Machine learning/Label and prediced label.csv')


for i, file in enumerate(sorted(os.listdir())):
    sample = file[:5]
    image = sitk.GetArrayFromImage(sitk.ReadImage(file))
    Total_pixel = np.sum(image > 75)
    time = int(file[7])
    group = y.loc[i,'Label']
    percentage = 1.0
    time_perf = 0
    if i == 0:
        pass
    else:
        if sample != file_list[i-1]['Sample']:
            if y.loc[i,'Label'] == 0:
                pass
            else:
                time_perf = 1
                temp = 56000
                percentage = Total_pixel / 56000
        else: 
            if y.loc[i,'Label'] == 0:
                pass
            else:
                if y.loc[i-1,'Label'] == 0 :
                    time_perf = 1
                    temp  = file_list[i-1]['Total pixel']
                    percentage = Total_pixel / temp
                    if percentage > 1:
                        temp  = Total_pixel
                        percentage = 1.0
                else:
                    time_perf = file_list[i-1]['Time_perf'] + 1
                    percentage = Total_pixel / temp
    
    if percentage < 0.675 and percentage > 0.31:
        group = 2
    if percentage < 0.31:
        group = 3
                    
    
    file_instance = {'Sample':sample, 'Time (week)':time, 'File name':file,
            'Total pixel': Total_pixel, 'Time_perf': time_perf, 
            "%_to_1st":percentage, 'Group':group}
    file_list.append(file_instance)

df = pd.DataFrame(file_list)
df = df[['Sample', 'Time (week)', 'File name','Total pixel', 'Time_perf', "%_to_1st", 'Group' ]]
df['Label'] = y['Label']
df['pred'] = y.iloc[:,11]
df.astype({'Label': 'int', 'pred':'int'})

df.head()
df['Total pixel'].hist(bins = 100)
df['%_to_1st'].hist(bins = 100)


plt.scatter(df['Time (week)'].values, df['%_to_1st'].values)

perfdf = df[df['Label'] == 1]
false_negative = perfdf[perfdf['pred'] != perfdf['Label']]

bins = np.linspace(0, 90000, 50)
plt.hist(perfdf['Total pixel'].values, bins, alpha = 0.5, label = 'Perforated')
plt.hist(df[df['Label'] == 0]['Total pixel'].values, bins, alpha = 0.5, label = 'Intact')
plt.hist(false_negative['Total pixel'].values, bins, alpha = 0.5, label = 'False negative')

plt.legend(loc='upper right')
plt.show()

cnn_df = df.groupby('Sample').count()
plt.figure()
for sample, data in cnn_df.iterrows():
    if data['Label'] < 3:
        continue
    else:
        seq = df[(df['Sample'] == sample)]
        plt.plot(seq['Time (week)'].values, seq['Total pixel'])

plt.show()

ob3_df = df[df['Sample'].isin(cnn_df[cnn_df['Time (week)'] >=3 ].index)]

hist = np.histogram(ob3_df[ob3_df['Label'] == 1]['%_to_1st'].values, bins = 50)
plt.plot(hist[1][:-1], hist[0])

bins = np.linspace(0, 1.0, 50)
plt.hist(ob3_df[ob3_df['Label'] == 1]['%_to_1st'], bins, alpha = 0.5, label = 'Perforated')
