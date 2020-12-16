import argparse
import time
import sys
sys.path.append('testRNN/src')
from utils import mkdir, delete_folder
from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from mnistTestSuite_adv_test import mnist_lstm_train, mnist_lstm_adv_test
from mnistTestSuite_backdoor_test import mnist_lstm_backdoor_test
from lipoTestSuite import lipo_lstm_train, lipo_lstm_test
from ucf101_vgg16_lstm_TestSuite import vgg16_lstm_train, vgg16_lstm_test
from record import record
import re

def main():
    
    parser = argparse.ArgumentParser(description='testing for recurrent neural networks')
    parser.add_argument('--model', dest='modelName', choices=['mnist', 'sentiment', 'lipo', 'ucf101'], default='sentiment')
    parser.add_argument('--TestCaseNum', dest='TestCaseNum', default='10000')
    parser.add_argument('--Mutation', dest='Mutation', choices=['random', 'genetic'], default='random')
    parser.add_argument('--CoverageStop', dest='CoverageStop', default='0.9')
    parser.add_argument('--threshold_SC', dest='threshold_SC', default='0.6')
    parser.add_argument('--threshold_BC', dest='threshold_BC', default='0.8')
    parser.add_argument('--symbols_TC', dest='symbols_TC', default='3')
    parser.add_argument('--seq', dest='seq', default='[70,89]')
    parser.add_argument('--mode', dest='mode', choices=['train', 'test'], default='test')
    parser.add_argument('--output', dest='filename', default='testRNN/log_folder/record.txt', help='')
    args=parser.parse_args()
    # seq:
    # mnist [4,24]
    # sentiment [400,499]
    # lipo [60,79]
    # ucf101 [0,10]

    modelName = args.modelName
    mode = args.mode
    filename = args.filename
    threshold_SC = args.threshold_SC
    threshold_BC = args.threshold_BC
    symbols_TC = args.symbols_TC
    seq = args.seq
    seq = re.findall(r"\d+\.?\d*", seq)
    Mutation = args.Mutation
    CoverageStop = args.CoverageStop
    TestCaseNum = args.TestCaseNum

    # record time
    r = record(filename,time.time())
    if modelName == 'sentiment': 
        if mode == 'train': 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, CoverageStop)

    elif modelName == 'mnist':
        if mode == 'train':
            mnist_lstm_train()
        elif mode == 'backdoor':
            mnist_lstm_backdoor_test(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, CoverageStop)
        else:
            mnist_lstm_adv_test(r, threshold_SC, threshold_BC, symbols_TC, seq, TestCaseNum, Mutation, CoverageStop)

    elif modelName == 'lipo':
        if mode == 'train':
            lipo_lstm_train()
        else:
            lipo_lstm_test(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, CoverageStop)

    elif modelName == 'ucf101':
        if mode == 'train':
            vgg16_lstm_train()
        else:
            vgg16_lstm_test(r, threshold_SC, threshold_BC, symbols_TC, seq, TestCaseNum, Mutation, CoverageStop)

    else: 
        print("Please specify a model from {sentiment, mnist, lipo, ucf101}")
    
    r.close()

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))