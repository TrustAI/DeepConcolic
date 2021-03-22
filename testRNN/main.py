#!/usr/bin/env python3
import argparse
import time
import sys
import os
__thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert (0, os.path.join (__thisdir, 'src'))
from utils import delete_folder
from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from mnistTestSuite_adv_test import mnist_lstm_train, mnist_lstm_adv_test
from mnistTestSuite_backdoor_test import mnist_lstm_backdoor_test
from ucf101_vgg16_lstm_TestSuite import vgg16_lstm_train, vgg16_lstm_test
from record import record
import re

def main():

    parser = argparse.ArgumentParser(description='testing for recurrent neural networks')
    parser.add_argument('--model', dest='modelName', choices=['mnist', 'fashion_mnist', 'sentiment', 'ucf101'], default='fashion_mnist')
    parser.add_argument('--TestCaseNum', dest='TestCaseNum', default='10000')
    parser.add_argument('--Mutation', dest='Mutation', choices=['random', 'genetic'], default='random')
    parser.add_argument('--CoverageStop', dest='CoverageStop', default='0.9')
    parser.add_argument('--threshold_SC', dest='threshold_SC', default='0.6')
    parser.add_argument('--threshold_BC', dest='threshold_BC', default='0.7')
    parser.add_argument('--symbols_TC', dest='symbols_TC', default='3')
    parser.add_argument('--seq', dest='seq', default='[400,499]')
    parser.add_argument('--mode', dest='mode', choices=['train', 'test'], default='test')
    parser.add_argument('--outputs', '--outdir', '-o', dest='outdir', default='testRNN_output', help='')
    parser.add_argument('--dataset', help='Test dataset file (in numpy persistent data format---for UCF101 only)', metavar = 'NP(Z)')
    args=parser.parse_args()
    # seq:
    # mnist [4,24]
    # sentiment [400,499]
    # lipo [60,79]
    # ucf101 [0,10]

    modelName = args.modelName
    mode = args.mode
    outdir = args.outdir
    dataset = args.dataset
    threshold_SC = args.threshold_SC
    threshold_BC = args.threshold_BC
    symbols_TC = args.symbols_TC
    seq = re.findall(r"\d+\.?\d*", args.seq)
    Mutation = args.Mutation
    CoverageStop = args.CoverageStop
    TestCaseNum = args.TestCaseNum

    if dataset is not None and \
       (not os.path.exists (dataset) or not os.access (dataset, os.R_OK)):
        sys.exit (f'Unreadable dataset file `{dataset}\'')

    r = None
    if mode != 'train':
        # record time
        r = record (os.path.join (outdir, "record.txt"), time.time())

        # reset output folder:
        delete_folder (r.subdir ('adv_output'))

    if modelName == 'sentiment':
        if mode == 'train':
            sentimentTrainModel()
        else:
            sentimentGenerateTestSuite(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, CoverageStop)

    elif modelName == 'mnist' or 'fashion_mnist':
        modelFile = os.path.join ('saved_models',
                                  'mnist_lstm.h5' if modelName == 'mnist' else \
                                  'fashion_mnist_lstm.h5')
        if mode == 'train':
            mnist_lstm_train (modelName, modelFile)
        else:
            routine = mnist_lstm_backdoor_test if mode == 'backdoor' else mnist_lstm_adv_test
            routine (r, threshold_SC, threshold_BC, symbols_TC, seq, TestCaseNum, Mutation,
                     modelName, modelFile)

    elif modelName == 'ucf101':
        if mode == 'train':
            vgg16_lstm_train()
        else:
            vgg16_lstm_test(r, dataset, threshold_SC, threshold_BC, symbols_TC, seq, TestCaseNum, Mutation, CoverageStop)

    else:
        print("Please specify a model from {sentiment, mnist, ucf101}")

    if r is not None:
        r.close()

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
