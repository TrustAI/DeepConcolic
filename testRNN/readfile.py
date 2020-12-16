import numpy as np
import scipy.io as scio
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Coverage Updating Plot')
parser.add_argument('--output', dest='filename', default='./log_folder/record.txt', help='')
parser.add_argument('--metrcis', dest='metrics', default='all', help='')
args=parser.parse_args()

filename = args.filename
metrics = args.metrics


lines = open(filename,'r').readlines()
nc = []
SC = []
BC = []
TC_p = []
TC_n = []
for i, line in enumerate(lines):
    parts = line.split(' ')
    if parts[0] == 'cell':
        SC.append(float(parts[2]))
    elif parts[0] == 'neuron':
        nc.append(float(parts[2]))
    elif parts[0] == 'gate':
        BC.append(float(parts[2]))
    elif parts[0] == 'positive':
        TC_p.append(float(parts[3]))
    elif parts[0] == 'negative':
        TC_n.append(float(parts[3]))

nc = np.array(nc)
SC = np.array(SC)
BC = np.array(BC)
TC_p = np.array(TC_p)
TC_n = np.array(TC_n)
# io.savemat('log_folder/coverage_count_NC.mat', {'coverage_count_NC': nc})
# io.savemat('log_folder/coverage_count_SC.mat', {'coverage_count_SC': SC})
# io.savemat('log_folder/coverage_count_BC.mat', {'coverage_count_BC': BC})
# io.savemat('log_folder/coverage_count_TC_P.mat', {'coverage_count_TC_P': TC_p})
# io.savemat('log_folder/coverage_count_TC_N.mat', {'coverage_count_TC_N': TC_n})

if metrics == 'SC' :
    plt.plot(SC)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Cell Coverage Updating')
    plt.savefig("log_folder/SC.jpg")
elif metrics == 'NC' :
    plt.plot(nc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Neuron Coverage Updating')
    plt.savefig("log_folder/nc.jpg")
elif metrics == 'BC' :
    plt.plot(BC)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Gate Coverage Updating')
    plt.savefig("log_folder/BC.jpg")
elif metrics == 'TCP' :
    plt.plot(TC_p)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Positive Sequence Coverage Updating')
    plt.savefig("log_folder/TCp.jpg")

elif metrics == 'TCN' :
    plt.plot(TC_n)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Negative Sequence Coverage Updating')
    plt.savefig("log_folder/TCn.jpg")

elif metrics == 'all' :
    plt.figure()
    plt.plot(SC)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Cell Coverage Updating')
    plt.savefig("log_folder/SC.jpg")

    plt.figure()
    plt.plot(nc)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Neuron Coverage Updating')
    plt.savefig("log_folder/nc.jpg")

    plt.figure()
    plt.plot(BC)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Gate Coverage Updating')
    plt.savefig("log_folder/BC.jpg")

    plt.figure()
    plt.plot(TC_p)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Positive Sequence Coverage Updating')
    plt.savefig("log_folder/TCp.jpg")

    plt.figure()
    plt.plot(TC_n)
    plt.xlabel('Test Cases')
    plt.ylabel('Coverage Rate')
    plt.title('Negative Sequence Coverage Updating')
    plt.savefig("log_folder/TCn.jpg")

else :
    print("Please specify a metrics to plot {NC, SC, BC, TCP, TCN, all}")

data1 = scio.loadmat('./log_folder/feature_count_SC.mat')
feature_count_SC = data1['feature_count_SC']
plt.figure()
plt.bar(range(len(feature_count_SC[0])),feature_count_SC[0])
plt.xlabel('Test Conditions')
plt.ylabel('Covering Times')
plt.title('Cell Coverage Test Conditions Count')
plt.savefig("log_folder/feature_count_SC.jpg")

data2 = scio.loadmat('./log_folder/feature_count_BC.mat')
feature_count_BC = data2['feature_count_BC']
plt.figure()
plt.bar(range(len(feature_count_BC[0])),feature_count_BC[0])
plt.xlabel('Test Conditions')
plt.ylabel('Covering Times')
plt.title('Gate Coverage Test Conditions Count')
plt.savefig("log_folder/feature_count_BC.jpg")
