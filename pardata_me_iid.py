import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from subword_nmt.apply_bpe import BPE
import tensorflow as tf
drug_col = 'DrugID'
protein_col = 'ProteinID'
label_col = 'Label'
# seq_list = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
############generate protein embedding dictionary
vocab_csv = pd.read_csv('vocab.csv')
values = vocab_csv['Values'].values
protein_dict = dict(zip(values,range(1,len(values)+1)))
print(protein_dict)
#seq_dict = {k : v+1 for v,k in enumerate(seq_list)}
csv1 = pd.read_csv('morgan_train.csv')
csv2 = pd.read_csv('morgan_valid.csv')
csv3 = pd.read_csv('morgan_test.csv')
final = []
for k in range(len(csv1)):
    l = [i for i in csv1['SMILES'][k]]
    final += l
for k in range(len(csv2)):
    l = [i for i in csv2['SMILES'][k]]
    final += l
for k in range(len(csv3)):
    l = [i for i in csv3['SMILES'][k]]
    final += l
kk = list(set(final))
kk_dict = {k:v+1 for v,k in enumerate(kk)}
print('drug dict : ',kk_dict)
print('drug dict length : ',len(kk_dict))
def encod_SMILES(seq,kk_dict):
    if pd.isnull(seq):
        return [0]
    else:
        return [kk_dict[a] for a in seq]
vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt,merges=-1,separator='')
def encodeSeq(seq,protein_dict):
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]

#read training data and operate on traning data so that we can get the final training data
def parse_data(dti_dir,drug_dir,protein_dir,my_matrix_dir,my_matrix_vec='iid',prot_vec='Convolution',my_matrix_len=21411,prot_len=800,drug_vec='Convolution',drug_len=2048,drug_len2=100):

    print("parsing data : {0},{1},{2}".format(dti_dir,drug_dir,protein_dir))
    dti_df = pd.read_csv(dti_dir)

    drug_df = pd.read_csv(drug_dir,index_col=drug_col)
    drug_df['drug_embedding'] = drug_df.SMILES.map(lambda a: encod_SMILES(a,kk_dict))

    protein_df = pd.read_csv(protein_dir,index_col=protein_col)
    protein_df['encoded_sequence'] = protein_df.Target_Sequence.map(lambda a: encodeSeq(a,protein_dict))

    dti_df = pd.merge(dti_df,drug_df,left_on=drug_col,right_index=True)

    dti_df = pd.merge(dti_df,protein_df,left_on=protein_col,right_index=True)
    #drug_feature = dti_df['morgan_fp'].values
    # drug_feature = np.stack(dti_df[drug_vec].map(lambda fp:fp.split('\t')))
    c_train = dti_df['morgan_fp'].values
    l = []
    for i in c_train:
      temp = [int(k) for k in i]
      l.append(temp)
    drug_feature = np.array(l)

    drug_feature2 = sequence.pad_sequences(dti_df['drug_embedding'].values,drug_len2,padding='post')
    protein_feature = sequence.pad_sequences(dti_df['encoded_sequence'].values, prot_len,padding='post')
    protein_feature2 = sequence.pad_sequences(dti_df['encoded_sequence'].values,prot_len,padding='post')
    #protein_feature2 = tf.cast(protein_feature2,dtype=tf.float32)
    label = [int(i) for i in dti_df[label_col].values]
    label = np.array(label)
    print('\t Positive data :   \t')
    print(sum(dti_df[label_col]))
    print('\t Negative data :   \t')
    print(dti_df.shape[0] - sum(dti_df[label_col]))
    #print('\t Positive data : %d' ,(sum(dti_df[label_col])))
    #print('\t Negative data : %d' ,(dti_df.shape[0]- sum(dti_df[label_col])))

    my_df = pd.read_csv(my_matrix_dir)
    # Drop accidental index columns commonly saved by pandas.
    drop_cols = [c for c in my_df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        my_df = my_df.drop(columns=drop_cols)

    my_matrix_me = my_df.values

    # Ensure IID features are float32.
    try:
        my_matrix_me = my_matrix_me.astype(np.float32)
    except Exception:
        my_matrix_me = np.array(my_matrix_me, dtype=np.float32)

    # Ensure 2D shape: (N, D)
    if my_matrix_me.ndim == 1:
        my_matrix_me = my_matrix_me.reshape(-1, 1)

    # Enforce expected dimensionality for the model input.
    if my_matrix_len is not None and int(my_matrix_len) > 0:
        target_dim = int(my_matrix_len)
        if my_matrix_me.shape[1] > target_dim:
            my_matrix_me = my_matrix_me[:, :target_dim]
        elif my_matrix_me.shape[1] < target_dim:
            pad_width = target_dim - my_matrix_me.shape[1]
            my_matrix_me = np.pad(my_matrix_me, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    return {
        "protein_feature" : protein_feature,
        "protein_feature2" : protein_feature2,
        "drug_feature" : drug_feature,
        "drug_feature2": drug_feature2,
        "my_matrix_feature" : my_matrix_me,
        "Label" : label,
    }
