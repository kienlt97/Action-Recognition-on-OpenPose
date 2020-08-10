import csv
time = []
pos = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvfile1 = csv.reader(csvfile, delimiter=' ')
        with open(filename.replace('.txt','.csv'), 'a+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in csvfile1:
                writer.writerow(row)
                
if __name__ =='__main__':
	get_data('origin_data.txt')