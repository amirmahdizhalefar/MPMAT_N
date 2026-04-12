if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #train data
    parser.add_argument('--dti_dir',default='train.csv')
    parser.add_argument('--drug_dir',default='morgan_train.csv')
    parser.add_argument('--protein_dir',default='protein_train.csv')
    parser.add_argument('--my_matrix_dir', default= 'my_matrix_train.csv')  # New argument for my_matrix_train
    #valid data
    parser.add_argument('--test-name','-n',default='data')
    parser.add_argument('--test-dti-dir','-i',default='valid.csv')
    parser.add_argument('--test-drug-dir','-d',default='morgan_valid.csv')
    parser.add_argument('--test-protein-dir','-t',default='protein_valid.csv')
    parser.add_argument('--test_my_matrix_dir', default= 'my_matrix_valid.csv')  # New argument for my_matrix_valid
    #struc params
    parser.add_argument('--window-sizes','-w',type=int,default=15)
    parser.add_argument('--protein-layers','-p',type=int,default=64)
    parser.add_argument('--drug-layers','-c',type=int,default=128)
    parser.add_argument('--fc-layers','-f',type=int,default=64)
    parser.add_argument('--my_matrix_layers', type=int, default=64)  # New argument for my_matrix_layers
    #training params
    parser.add_argument('--learning-rate','-r',default=0.0001,type=float)
    parser.add_argument('--n-epoch','-e',default=38,type=int)
    #type params
    parser.add_argument('--prot-vec','-v',default='Convolution')
    parser.add_argument('--prot-len','-l',default=800,type=int)
    parser.add_argument('--drug-vec','-V',default='morgan_fp')
    parser.add_argument('--drug-len','-L',default=2048,type=int)
    parser.add_argument('--drug-len2',default=100,type=int)
    parser.add_argument('--my_matrix-vec',default='Convolution')
    parser.add_argument('--my_matrix_len', default=21411, type=int)  # New argument for my_matrix_len
    #other params
    parser.add_argument('--activation','-a',default='relu',type=str)
    parser.add_argument('--dropout','-D',default=0.2,type=float)
    parser.add_argument('--n-filters','-F',default=128,type=int)
    parser.add_argument('--batch-size','-b',type=int,default=8)
    parser.add_argument('--decay','-y',default=0.0001,type=float)
    #mode params
    parser.add_argument('--validation',action='store_true')
    args = parser.parse_args()
    #traindata  dic
    traindata_dic = {
        'dti_dir': args.dti_dir,
        'drug_dir': args.drug_dir,
        'protein_dir': args.protein_dir,
        'my_matrix_dir': args.my_matrix_dir,  # Load and preprocess my_matrix_train      
    }
    # my_matrix_dic = {
    #     'train_my_matrix':args.train_my_matrix # Load and preprocess my_matrix_train 
    # }
    #pack the test  datasets
    testnames = args.test_name,
    validdata_dic = {
        'test_dti_dir': args.test_dti_dir,
        'test_drugs_dir': args.test_drug_dir,
        'test_proteins_dir': args.test_protein_dir,
        'test_my_matrix_dir': args.test_my_matrix_dir
    }
    # test_sets = zip(testnames,test_dti,test_drugs,test_proteins,test_my_matrix)
    print(traindata_dic['drug_dir'])