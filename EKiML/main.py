import argparse
import sys
sys.path.append('EKiML/src')
from embedding_knowledge import embedding_knowledge
from synthesis_knowledge import synthesis_knowledge

def main():
    
    parser = argparse.ArgumentParser(description='Embedding Knowledge into Random Forest')
    parser.add_argument('--Dataset', dest='Dataset', default='iris', help='')
    parser.add_argument('--Mode', dest='Mode', default='embedding', help='')
    parser.add_argument('--Embedding_Method', dest='Embedding_Method', default='white-box', help='')
    parser.add_argument('--Model', dest='Model', default='forest', help='')
    parser.add_argument('--Pruning', dest='Pruning', default = 'False', help='')
    parser.add_argument('--SaveModel', dest='SaveModel', default = 'True', help='')
    parser.add_argument('--output', dest='filename', default='EKiML/model/', help='')
    args=parser.parse_args()

    # dataset: iris, breast_cancer, mushroom, nursery, cod-rna, sensorless, mnist
    # check more details about dataset in UCI Dataset
    dataset = args.Dataset
    embedding = args.Embedding_Method
    model = args.Model
    pruning = args.Pruning
    save_model = args.SaveModel
    mode = args.Mode
    filename = args.filename

    if mode == 'embedding':
        embedding_knowledge(dataset, embedding, model, pruning, save_model, filename)
    elif mode == 'synthesis':
        synthesis_knowledge(dataset, embedding, filename)
    else:
        "please specify the embedding method, black-box settings or white-box setting ?"


if __name__ == "__main__":
    main()