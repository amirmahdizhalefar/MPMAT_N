"""
main.py  –  MPMAT version
==========================
Protein (steps 1, 4) and Drug (steps 2, 3) branches are IDENTICAL to the
original Multi-TransDTI code.  The only additions are:

  • Branch 1: meta-path MLP+Transformer encoder (Section II-E-2)
  • Step 5 fusion: Concatenate([zm, finalmodel_D, finalmodel_P]) → 256 dims
  • Model inputs now include metapath_input
  • validation() threads metapath_feature through fit/predict calls
  • CLI gains --metapath-cache-dir argument

Do NOT touch the protein or drug branches below – they are marked UNCHANGED.
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Concatenate, Convolution1D, GlobalMaxPooling1D,
                                      GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D)
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, Activation,
                                      Dropout, Embedding, SpatialDropout1D)
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
from pardata import parse_data, get_metapath_len
from transformer import Transformer
from metapath_encoder import build_metapath_encoder   # NEW
import pandas as pd


class Net(object):

    # ── Unchanged helper from original code ───────────────────────────────────
    def Player(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same',
                                    kernel_initializer=initializer,
                                    kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    # ─────────────────────────────────────────────────────────────────────────
    def modelvv(self, dropout, drug_layers, protein_strides, filters, fc_layers,
                prot_vec=False, prot_len=2500, activation='relu',
                protein_layers=None, initializer='glorot_normal',
                drug_len=2048, drug_vec='ECFP4', drug_len2=100,
                metapath_len=3759):             # ← NEW parameter

        def return_tuple(value):
            if type(value) is int:
                return [value]
            else:
                return tuple(value)

        regularizer_param = 0.001
        params_dict = {
            'kernel_initializer': initializer,
            'kernel_regularizer': l2(regularizer_param)
        }

        # ════════════════════════════════════════════════════════════════════
        # BRANCH 1 – Meta-path MLP+Transformer  (NEW – Section II-E-2)
        # ════════════════════════════════════════════════════════════════════
        metapath_input, zm = build_metapath_encoder(
            metapath_len=metapath_len,   # 3 × Nt  (e.g. 3759)
            dmodel=128,                  # zm ∈ R^128
            num_heads=4,                 # dk = 128/4 = 32
            dff=256,                     # FFN hidden dim
            num_layers=2,                # LTR = 2 Transformer encoder layers
            dropout_rate=dropout,
            mlp_dims=(512, 256),         # MLP schedule: 3Nt→512→256→128
            reg=regularizer_param,
        )
        # zm: (batch, 128)

        # ════════════════════════════════════════════════════════════════════
        # STEP 1 – Protein Transformer  (UNCHANGED)
        # ════════════════════════════════════════════════════════════════════
        num_layers_tr = 2
        model_size    = 20
        num_heads_tr  = 5
        dff_size      = 64
        maxlen        = 800
        vocab_size    = 474

        enc_inputs = keras.layers.Input(shape=(maxlen,))
        dec_inputs = keras.layers.Input(shape=(maxlen,))
        transformer = Transformer(num_layers=num_layers_tr, model_size=model_size,
                                  num_heads=num_heads_tr, dff_size=dff_size,
                                  vocab_size=vocab_size + 1, maxlen=maxlen)
        final_output = transformer([enc_inputs, dec_inputs])
        final_output = SpatialDropout1D(0.2)(final_output)
        final_output = Convolution1D(filters=128, kernel_size=15, padding='same',
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=l2(0.001))(final_output)
        final_output = Activation('relu')(final_output)
        final_output = GlobalMaxPooling1D()(final_output)
        final_output = Dense(64, 'relu', **params_dict)(final_output)

        # ════════════════════════════════════════════════════════════════════
        # STEP 2 – Drug Morgan Fingerprints  (UNCHANGED)
        # ════════════════════════════════════════════════════════════════════
        input_d    = Input(shape=(drug_len,))
        drug_layers_t = return_tuple(drug_layers)
        for layer_size in drug_layers_t:
            model_d = Dense(layer_size, **params_dict)(input_d)
            model_d = BatchNormalization()(model_d)
            model_d = Activation(activation)(model_d)
            model_d = Dropout(dropout)(model_d)

        # ════════════════════════════════════════════════════════════════════
        # STEP 3 – Drug SMILES Conv1D  (UNCHANGED)
        # ════════════════════════════════════════════════════════════════════
        input_d2 = Input(shape=(drug_len2,))
        model_d2 = Embedding(42, 10, embeddings_initializer=initializer,
                             embeddings_regularizer=l2(regularizer_param))(input_d2)
        model_d2 = SpatialDropout1D(0.2)(model_d2)
        model_d2 = [self.Player(10, 128, activation, initializer, regularizer_param)(model_d2)]
        if len(model_d2) != 1:
            model_d2 = Concatenate(axis=1)(model_d2)
        else:
            model_d2 = model_d2[0]
        protein_layers_t = return_tuple(protein_layers)
        for protein_layer in protein_layers_t:
            model_d2 = Dense(64, **params_dict)(model_d2)
            model_d2 = BatchNormalization()(model_d2)
            model_d2 = Activation(activation)(model_d2)
            model_d2 = Dropout(dropout)(model_d2)

        # ════════════════════════════════════════════════════════════════════
        # STEP 4 – Protein Conv1D  (UNCHANGED)
        # ════════════════════════════════════════════════════════════════════
        input_p = Input(shape=(prot_len,))
        model_p = Embedding(vocab_size + 1, 20, embeddings_initializer=initializer,
                            embeddings_regularizer=l2(regularizer_param))(input_p)
        model_p = SpatialDropout1D(0.2)(model_p)
        protein_strides_t = return_tuple(protein_strides)
        model_p = [self.Player(stride_size, filters, activation, initializer,
                               regularizer_param)(model_p)
                   for stride_size in protein_strides_t]
        if len(model_p) != 1:
            model_p = Concatenate(axis=1)(model_p)
        else:
            model_p = model_p[0]
        for protein_layer in protein_layers_t:
            model_p = Dense(64, **params_dict)(model_p)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            model_p = Dropout(dropout)(model_p)

        # ════════════════════════════════════════════════════════════════════
        # STEP 5 – Fusion  (MODIFIED: [zm ; finalmodel_D ; finalmodel_P])
        # Original concatenated [finalmodel_D, finalmodel_P] → 128 dims.
        # MPMAT concatenates [zm, finalmodel_D, finalmodel_P] → 256 dims.
        # Eq. 20: z = [zm ; zp ; zd]
        # ════════════════════════════════════════════════════════════════════
        finalmodel_D = Concatenate(axis=1)([model_d, model_d2])
        finalmodel_D = Dense(64, **params_dict)(finalmodel_D)   # zd ∈ R^64

        finalmodel_P = Concatenate(axis=1)([model_p, final_output])
        finalmodel_P = Dense(64, **params_dict)(finalmodel_P)   # zp ∈ R^64

        # Multi-modal fusion: z = [zm(128); zp(64); zd(64)] = 256 dims
        model_ttt = Concatenate(axis=1)([zm, finalmodel_D, finalmodel_P])

        fc_layers_t = return_tuple(fc_layers)
        for fc_layer in fc_layers_t:
            model_ttt = Dense(units=fc_layer, **params_dict)(model_ttt)
            model_ttt = Activation(activation)(model_ttt)

        model_ttt = Dense(1, activation='sigmoid',
                          activity_regularizer=l2(regularizer_param),
                          **params_dict)(model_ttt)

        # Model inputs: metapath_input added (position 0)
        model_final = Model(
            inputs=[input_d, input_d2, input_p, enc_inputs, dec_inputs,
                    metapath_input],        # ← NEW input
            outputs=model_ttt
        )
        return model_final

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, dropout=0.2, drug_layers=512, protein_strides=15,
                 filters=64, learning_rate=0.0001, decay=0.0, drug_len2=100,
                 fc_layers=None, prot_vec=None, prot_len=2500, activation='relu',
                 drug_len=2048, drug_vec='ECFP4', protein_layers=None,
                 metapath_len=3759,            # ← NEW
                 metapath_cache_dir=None):     # ← NEW (stored for validation)

        self.__dropout           = dropout
        self.__drugs_layer       = drug_layers
        self.__protein_strides   = protein_strides
        self.__filters           = filters
        self.__learning_rate     = learning_rate
        self.__decay             = decay
        self.__fc_layers         = fc_layers
        self.__prot_vec          = prot_vec
        self.__prot_len          = prot_len
        self.__activation        = activation
        self.__drug_len          = drug_len
        self.__drug_vec          = drug_vec
        self.__prot_layers       = protein_layers
        self.__drug_len2         = drug_len2
        self.__metapath_len      = metapath_len          # NEW
        self.__metapath_cache_dir = metapath_cache_dir  # NEW

        self.__model_t = self.modelvv(
            self.__dropout, self.__drugs_layer, self.__protein_strides,
            self.__filters, self.__fc_layers,
            prot_vec=self.__prot_vec, prot_len=self.__prot_len,
            activation=self.__activation, protein_layers=self.__prot_layers,
            drug_vec=self.__drug_vec, drug_len=self.__drug_len,
            drug_len2=self.__drug_len2,
            metapath_len=self.__metapath_len,            # NEW
        )

        opt = Adam(learning_rate=learning_rate)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy',
                               metrics=['accuracy'])

    # ─────────────────────────────────────────────────────────────────────────
    def summary(self):
        self.__model_t.summary()

    # ── Helpers to assemble model input lists ─────────────────────────────────

    def _inputs(self, drug_feature, drug_feature2, protein_feature,
                protein_feature2, metapath_feature):
        """Return the 6-element input list expected by the model."""
        return [drug_feature, drug_feature2, protein_feature,
                protein_feature2, protein_feature2,
                metapath_feature]

    # ─────────────────────────────────────────────────────────────────────────
    def validation(self, drug_feature, drug_feature2, protein_feature,
                   protein_feature2, Label, metapath_feature,   # ← NEW
                   n_epoch=10, batch_size=32, **kwargs):

        auc_temp  = 0
        auclist   = []
        auprlist  = []

        for i in range(n_epoch):
            self.__model_t.fit(
                self._inputs(drug_feature, drug_feature2,
                             protein_feature, protein_feature2,
                             metapath_feature),
                Label,
                epochs=i + 1, verbose=1, initial_epoch=i,
                batch_size=batch_size, shuffle=True
            )
            for dataset in kwargs:
                print("\tPrediction of " + dataset)

                test_p   = kwargs[dataset]["protein_feature"]
                test_p2  = kwargs[dataset]["protein_feature2"]
                test_d   = kwargs[dataset]["drug_feature"]
                test_d2  = kwargs[dataset]['drug_feature2']
                test_mp  = kwargs[dataset]['metapath_feature']   # NEW
                test_label = kwargs[dataset]["Label"]

                prediction = self.__model_t.predict(
                    self._inputs(test_d, test_d2, test_p, test_p2, test_mp))

                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC   = auc(fpr, tpr)
                precision, recall, thresholds_AUPR = precision_recall_curve(
                    test_label, prediction)
                AUPR  = auc(recall, precision)

                distance = [tpr[k] - fpr[k] for k in range(len(tpr))]
                opt_AUC  = thresholds_AUC[distance.index(max(distance))]

                auclist.append(AUC)
                auprlist.append(AUPR)

                if AUC > auc_temp:
                    print('valid AUC : ', AUC,
                          '  ******************** test data')
                    auc_temp = AUC

                    # ── Evaluate on held-out test set ─────────────────────
                    testcsv      = 'test.csv'
                    morgantest   = 'morgan_test.csv'
                    proteintest  = 'protein_test.csv'
                    finalsets    = parse_data(
                        testcsv, morgantest, proteintest,
                        metapath_cache_dir=self.__metapath_cache_dir)
                    print(finalsets)

                    drug_fea    = finalsets['drug_feature']
                    drug_fea2   = finalsets['drug_feature2']
                    protein_fea = finalsets['protein_feature']
                    protein_fea2= finalsets['protein_feature2']
                    test_mp_fin = finalsets['metapath_feature']   # NEW
                    lab         = finalsets['Label']

                    final_predition = self.__model_t.predict(
                        self._inputs(drug_fea, drug_fea2,
                                     protein_fea, protein_fea2, test_mp_fin))

                    finalfpr, finaltpr, finalthresholds = roc_curve(
                        lab, final_predition)
                    finalprecision, finalrecall, finalthres = \
                        precision_recall_curve(lab, final_predition)
                    finalAUC  = auc(finalfpr, finaltpr)
                    finalAUPR = auc(finalrecall, finalprecision)

                    dis = [finaltpr[k] - finalfpr[k] for k in range(len(finaltpr))]
                    opt = finalthresholds[dis.index(max(dis))]

                    y_preddd = [1 if final_predition[k] >= opt else 0
                                for k in range(len(final_predition))]
                    confusion_matix = confusion_matrix(lab, y_preddd)
                    ACC   = ((confusion_matix[0][0] + confusion_matix[1][1]) /
                             (confusion_matix[0][0] + confusion_matix[0][1] +
                              confusion_matix[1][0] + confusion_matix[1][1]))
                    F1    = f1_score(lab, y_preddd)
                    Sensi = (confusion_matix[0][0] /
                             (confusion_matix[0][0] + confusion_matix[0][1]))
                    Speci = (confusion_matix[1][1] /
                             (confusion_matix[1][1] + confusion_matix[1][0]))

                    print('threshold_AUC', opt)
                    print('ACC : ',         ACC)
                    print('AUC : ',         finalAUC)
                    print('AUPR : ',        finalAUPR)
                    print('Sensitivity : ', Sensi)
                    print('Specificity : ', Speci)
                    print('F1 score : ',    F1)

                    out1 = open('1.csv', 'w')
                    out1.write('fpr,tpr\n')
                    for k in range(len(finalfpr)):
                        out1.write(str(finalfpr[k]) + ',' +
                                   str(finaltpr[k]) + '\n')
                    out1.close()

                    out2 = open('2.csv', 'w')
                    out2.write('recall,precision\n')
                    for k in range(len(finalprecision)):
                        out2.write(str(finalrecall[k]) + ',' +
                                   str(finalprecision[k]) + '\n')
                    out2.close()

                y_pred = [1 if prediction[k] >= opt_AUC else 0
                          for k in range(len(prediction))]
                confusion_matix = confusion_matrix(test_label, y_pred)
                ACC   = ((confusion_matix[0][0] + confusion_matix[1][1]) /
                         (confusion_matix[0][0] + confusion_matix[0][1] +
                          confusion_matix[1][0] + confusion_matix[1][1]))
                Sensi = (confusion_matix[0][0] /
                         (confusion_matix[0][0] + confusion_matix[0][1]))
                Speci = (confusion_matix[1][1] /
                         (confusion_matix[1][1] + confusion_matix[1][0]))
                F1    = f1_score(test_label, y_pred)

                print('\n\n\n')
                print("\t \t  ACC:\t  ",         ACC)
                print('\t optimal threshold:  \t ', opt_AUC)
                print("\t \t  AUC:\t  ",         AUC)
                print("\t \t AUPR:\t  ",         AUPR)
                print("\t \t  Sensitivity:\t  ", Sensi)
                print("\t \t  Specificity:\t  ", Speci)
                print("\t \t  F1_score:\t  ",    F1)
                print("=================================================")

    def save(self, output_file):
        self.__model_t.save(output_file)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # train data
    parser.add_argument('--dti_dir',     default='train.csv')
    parser.add_argument('--drug_dir',    default='morgan_train.csv')
    parser.add_argument('--protein_dir', default='protein_train.csv')
    # valid data
    parser.add_argument('--test-name',         '-n', default='data')
    parser.add_argument('--test-dti-dir',      '-i', default='valid.csv')
    parser.add_argument('--test-drug-dir',     '-d', default='morgan_valid.csv')
    parser.add_argument('--test-protein-dir',  '-t', default='protein_valid.csv')
    # structural params
    parser.add_argument('--window-sizes',   '-w', type=int, default=15)
    parser.add_argument('--protein-layers', '-p', type=int, default=64)
    parser.add_argument('--drug-layers',    '-c', type=int, default=128)
    parser.add_argument('--fc-layers',      '-f', type=int, default=64)
    # training params
    parser.add_argument('--learning-rate',  '-r', default=0.0001, type=float)
    parser.add_argument('--n-epoch',        '-e', default=58,      type=int)
    # type params
    parser.add_argument('--prot-vec',  '-v', default='Convolution')
    parser.add_argument('--prot-len',  '-l', default=800,  type=int)
    parser.add_argument('--drug-vec',  '-V', default='morgan_fp')
    parser.add_argument('--drug-len',  '-L', default=2048, type=int)
    parser.add_argument('--drug-len2',       default=100,  type=int)
    # other params
    parser.add_argument('--activation', '-a', default='relu', type=str)
    parser.add_argument('--dropout',    '-D', default=0.2,    type=float)
    parser.add_argument('--n-filters',  '-F', default=128,    type=int)
    parser.add_argument('--batch-size', '-b', default=32,     type=int)
    parser.add_argument('--decay',      '-y', default=0.0001, type=float)
    # ── NEW: meta-path cache directory ───────────────────────────────────────
    parser.add_argument('--metapath-cache-dir', default='metapath_cache',
                        help='Path to directory created by compute_metapaths.py')
    args = parser.parse_args()

    # ── Resolve metapath_len from cache ──────────────────────────────────────
    metapath_len = get_metapath_len(args.metapath_cache_dir)
    print(f"  metapath_len (3·Nt) = {metapath_len}")

    # ── Type-params dict (unchanged) ─────────────────────────────────────────
    type_params_dict = {
        'prot_vec':  args.prot_vec,
        'prot_len':  args.prot_len,
        'drug_vec':  args.drug_vec,
        'drug_len':  args.drug_len,
        'drug_len2': args.drug_len2,
    }

    # ── Network-params dict (unchanged + metapath_len) ───────────────────────
    network_params = {
        'drug_layers':      args.drug_layers,
        'protein_strides':  args.window_sizes,
        'protein_layers':   args.protein_layers,
        'fc_layers':        args.fc_layers,
        'learning_rate':    args.learning_rate,
        'decay':            args.decay,
        'activation':       args.activation,
        'filters':          args.n_filters,
        'dropout':          args.dropout,
        'metapath_len':     metapath_len,                  # NEW
        'metapath_cache_dir': args.metapath_cache_dir,    # NEW
    }
    network_params.update(type_params_dict)

    print('\t model parameters summary \t')
    print('=====================================')
    for key in network_params.keys():
        print('{:25s}:  {:10s}'.format(key, str(network_params[key])))

    dti_prediction_model = Net(**network_params)
    dti_prediction_model.summary()

    # ── Parse training data (with metapath) ──────────────────────────────────
    traindata_dic = {
        'dti_dir':     args.dti_dir,
        'drug_dir':    args.drug_dir,
        'protein_dir': args.protein_dir,
        'metapath_cache_dir': args.metapath_cache_dir,    # NEW
    }
    traindata_dic.update(type_params_dict)
    traindata_dic = parse_data(**traindata_dic)

    # ── Parse validation/test data (with metapath) ───────────────────────────
    testnames    = (args.test_name,)
    test_dti     = (args.test_dti_dir,)
    test_drugs   = (args.test_drug_dir,)
    test_proteins= (args.test_protein_dir,)
    test_sets    = zip(testnames, test_dti, test_drugs, test_proteins)

    test_dic = {
        test_name: parse_data(
            t_dti, t_drug, t_protein,
            metapath_cache_dir=args.metapath_cache_dir,   # NEW
            **type_params_dict)
        for test_name, t_dti, t_drug, t_protein in test_sets
    }

    # ── Training params ───────────────────────────────────────────────────────
    training_params_dict = {
        'n_epoch':    args.n_epoch,
        'batch_size': args.batch_size,
    }

    print("\tvalidation summary\t")
    print("=====================================================")
    for key in training_params_dict.keys():
        print("{:20s} : {:10s}".format(key, str(training_params_dict[key])))
    print("=====================================================")

    validation_params = {}
    validation_params.update(training_params_dict)
    validation_params.update(traindata_dic)
    validation_params.update(test_dic)

    dti_prediction_model.validation(**validation_params)
