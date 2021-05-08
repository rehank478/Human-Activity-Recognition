from csv import writer
import csv

with open("data_2_s2TxLos.csv", 'w', newline='') as write_obj:
    csv_writer = writer(write_obj)
    with open('15sec_2_s2TxLos.txt','rt')as f:
        for t in range(0,6):
            next(f)
        data = csv.reader(f)
        ct = 0
        for i in data:
            ct += 1
            if len(i)==ct:
                break
            if len(i)==0:
                continue
            if i[0]=='CSI_DATA':
                csv_writer.writerow(i)
            