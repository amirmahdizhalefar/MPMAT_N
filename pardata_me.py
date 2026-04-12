import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from subword_nmt.apply_bpe import BPE
import tensorflow as tf

drug_col = 'DrugID'
protein_col = 'ProteinID'
label_col = 'Label'

# ---------- protein embedding dictionary ----------
vocab_csv = pd.read_csv('vocab.csv')
values = vocab_csv['Values'].values
protein_dict = dict(zip(values, range(1, len(values) + 1)))
print(protein_dict)

# ---------- SMILES character dictionary ----------
# FIX 1: load only the SMILES column to avoid the overflow on morgan_fp
csv1 = pd.read_csv('morgan_train.csv', usecols=['SMILES'])
csv2 = pd.read_csv('morgan_valid.csv', usecols=['SMILES'])
csv3 = pd.read_csv('morgan_test.csv', usecols=['SMILES'])

final = []
for df in [csv1, csv2, csv3]:
    for smiles in df['SMILES']:
        if pd.notnull(smiles):
            final += list(smiles)

kk = list(set(final))
kk_dict = {k: v + 1 for v, k in enumerate(kk)}
print('drug dict : ', kk_dict)
print('drug dict length : ', len(kk_dict))


def encod_SMILES(seq, kk_dict):
    if pd.isnull(seq):
        return [0]
    return [kk_dict[a] for a in seq]


vocab_txt = open('vocab.txt')
bpe = BPE(vocab_txt, merges=-1, separator='')


def encodeSeq(seq, protein_dict):
    firststep = bpe.process_line(seq).split()
    return [protein_dict[a] for a in firststep]


def parse_data(dti_dir, drug_dir, protein_dir, my_matrix_dir,
               my_matrix_vec='Convolution', prot_vec='Convolution',
               my_matrix_len=21411, prot_len=800,
               drug_vec='Convolution', drug_len=2048, drug_len2=100):

    print("parsing data : {0},{1},{2}".format(dti_dir, drug_dir, protein_dir))
    dti_df = pd.read_csv(dti_dir)

    # FIX 2: force morgan_fp to be read as a plain string so its long
    # binary value is never interpreted as a numeric type
    drug_df = pd.read_csv(drug_dir, index_col=drug_col, dtype={'morgan_fp': str})
    drug_df['drug_embedding'] = drug_df['SMILES'].map(
        lambda a: encod_SMILES(a, kk_dict))

    protein_df = pd.read_csv(protein_dir, index_col=protein_col)
    protein_df['encoded_sequence'] = protein_df['Target_Sequence'].map(
        lambda a: encodeSeq(a, protein_dict))

    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)

    # morgan_fp is now a string of '0'/'1' characters — iterate directly
    drug_feature = np.array([
        [int(bit) for bit in fp]
        for fp in dti_df['morgan_fp'].values
    ])

    drug_feature2 = sequence.pad_sequences(
        dti_df['drug_embedding'].values, drug_len2, padding='post')
    protein_feature = sequence.pad_sequences(
        dti_df['encoded_sequence'].values, prot_len, padding='post')
    protein_feature2 = sequence.pad_sequences(
        dti_df['encoded_sequence'].values, prot_len, padding='post')

    label = np.array([int(i) for i in dti_df[label_col].values])

    print('\t Positive data : ', sum(dti_df[label_col]))
    print('\t Negative data : ', dti_df.shape[0] - sum(dti_df[label_col]))

    my_matrix_me = pd.read_csv(my_matrix_dir).values

    return {
        "protein_feature":  protein_feature,
        "protein_feature2": protein_feature2,
        "drug_feature":     drug_feature,
        "drug_feature2":    drug_feature2,
        "my_matrix_feature": my_matrix_me,
        "Label":            label,
    }