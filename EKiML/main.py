import argparse
import sys
import os
__thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert (0, os.path.join (__thisdir, 'src'))
from embedding_knowledge import embedding_knowledge
from synthesis_knowledge import synthesis_knowledge

def ensure_directory (workdir, check_writable = False):
  if not os.path.exists (workdir):
    os.makedirs (workdir)
  if not os.path.isdir (workdir):
    raise ValueError (f'Expected directory {workdir}')
  if not os.access (workdir, os.R_OK):
    raise ValueError (f'Expected readable directory {workdir}')
  if check_writable and not os.access (workdir, os.W_OK):
    raise ValueError (f'Expected writable directory {workdir}')
  return workdir

def main():

    parser = argparse.ArgumentParser(description='Embedding Knowledge into Random Forest')

    parser.add_argument('--Datadir', dest='datadir', default=os.path.join ('EKiML', 'dataset'),
                        help='Datasets directory')
    parser.add_argument('--Dataset', dest='Dataset', default='har', help='')
    parser.add_argument('--Mode', dest='Mode', default='synthesis', help='')
    parser.add_argument('--Embedding_Method', dest='Embedding_Method', default='black-box', help='')
    parser.add_argument('--Model', dest='Model', default='tree', help='')
    parser.add_argument('--Pruning', dest='Pruning', default = 'False', help='')
    parser.add_argument('--SaveModel', dest='SaveModel', default = 'True', help='')
    parser.add_argument('--workdir', '--output', dest='workdir',
                        default='EKiML_workdir', help='Working directory')
    args=parser.parse_args()


    # dataset: iris, breast_cancer, har, mushroom, nursery, cod-rna, sensorless, mnist
    # check more details about dataset in UCI Dataset
    dataset = args.Dataset
    embedding = args.Embedding_Method
    model = args.Model
    pruning = args.Pruning
    save_model = args.SaveModel
    mode = args.Mode
    datadir = ensure_directory (args.datadir)
    workdir = ensure_directory (args.workdir, check_writable = mode == 'synthesis')

    if mode == 'embedding':
        embedding_knowledge(dataset, embedding, model, pruning, save_model, workdir,
                            datadir = datadir)
    elif mode == 'synthesis':
        synthesis_knowledge(dataset, embedding, model, workdir,
                            datadir = datadir)
    else:
        "please specify the embedding method, black-box settings or white-box setting ?"


if __name__ == "__main__":
    main()
