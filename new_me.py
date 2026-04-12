from tensorflow.keras.optimizers import Adam from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Concatenate,Convolution1D,GlobalMaxPooling1D,GlobalAveragePooling1D,MaxPooling1D,Max Pooling2D

from tensorflow.keras.layers import Input,Dense,BatchNormalization,Activation,Dropout,Embedding,SpatialDropout1D

import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import precision_recall_curve,auc,roc_curve,f1_score from sklearn.metrics import confusion_matrix
import numpy as np

from pardata_me import parse_data from transformer import Transformer
from tensorflow.keras.utils import plot_model import pandas as pd


# physical_devices = tf.config.list_physical_devices('GPU')

# tf.config.experimental.set_memory_growth(physical_devices, True)



class Net(object):

def Player(self,size,filters,activation,initializer,regularizer_param): def f(input):
model_p = Convolution1D(filters=filters,kernel_size=size,padding='same',kernel_initializer=initializer,

kernel_regularizer=l2(regularizer_param))(input) model_p = BatchNormalization()(model_p)
model_p = Activation(activation)(model_p) return GlobalMaxPooling1D()(model_p)
return f

# def modelvv(self,dropout,drug_layers,protein_strides,filters,fc_layers,prot_vec=False,prot_len= 2500,activation='relu',

#	protein_layers=None,initializer='glorot_normal',drug_len= 2048,drug_vec='ECFP4',drug_len2=100):

def modelvv(self,dropout,drug_layers,protein_strides,filters,fc_layers,prot_vec=False,prot_len= 2500,activation='relu',

protein_layers=None,initializer='glorot_normal',drug_len= 2048,drug_vec='ECFP4',drug_len2=100,

my_matrix_len=21411, my_matrix_layers=None, my_matrix_vec=False):




# define some parameters of the model def return_tuple(value):
if type(value) is int: return [value]
else:

return tuple(value) regularizer_param = 0.001 params_dict = {

'kernel_initializer' : initializer, 'kernel_regularizer' : l2(regularizer_param)
}








# #####00000000000000000000	step 0 my_matrix_ transformer # my_matrix_num_layers = 2
# my_matrix_model_size = 20 # my_matrix_num_heads = 5 # my_matrix_dff_size = 64
# my_matrix_maxlen = 21411 # my_matrix_vocab_size = 474
# my_matrix_enc_inputs = keras.layers.Input(shape=(my_matrix_maxlen,)) # my_matrix_dec_inputs = keras.layers.Input(shape=(my_matrix_maxlen,))
# my_matrix_transformer = Transformer(num_layers= my_matrix_num_layers, model_size= my_matrix_model_size, num_heads= my_matrix_num_heads, dff_size= my_matrix_dff_size,
#	vocab_size= my_matrix_vocab_size+1, maxlen= my_matrix_maxlen) # my_matrix_final_output = my_matrix_transformer([my_matrix_enc_inputs,
my_matrix_dec_inputs])

# my_matrix_final_output = SpatialDropout1D(0.2)( my_matrix_final_output)

# my_matrix_final_output = Convolution1D(filters=128,kernel_size=15, padding='same', kernel_initializer='glorot_normal',

#	kernel_regularizer=l2(0.001))( my_matrix_final_output)

# my_matrix_final_output = Activation('relu')( my_matrix_final_output)

# my_matrix_final_output = GlobalMaxPooling1D()(my_matrix_final_output) # #final_output = Dropout(dropout)(final_output)
# #final_output = MaxPooling1D()(final_output)

# my_matrix_final_output = Dense(64,'relu',**params_dict)( my_matrix_final_output)



#####00000000000000000000	step 0 my_matrix_ transformer convolution

my_matrix_num_layers = 2

my_matrix_model_size = 20

my_matrix_num_heads = 5

my_matrix_dff_size = 64

my_matrix_maxlen = 21411

my_matrix_vocab_size = 474



###################4444444444444444 step 4	define protein convolution my_matrix_enc_inputs = keras.layers.Input(shape=(my_matrix_maxlen,))
# input_MME = my_matrix_enc_inputs

MME_model = Embedding(my_matrix_vocab_size+ 1,20,embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param)) (my_matrix_enc_inputs)

MME_model = SpatialDropout1D(0.2)(MME_model) MME_strides = return_tuple(protein_strides)
MME_model = [self.Player(stride_size,filters,activation,initializer,regularizer_param) (MME_model) for stride_size in MME_strides]

if len(MME_model) != 1:

MME_model = Concatenate(axis=1)(MME_model) else:
MME_model = MME_model[0]

MME_layers = return_tuple(my_matrix_layers) for my_matrix_layer in MME_layers:
MME_model = Dense(256,**params_dict)(MME_model) MME_model = BatchNormalization()(MME_model) MME_model = Activation(activation)(MME_model) MME_model = Dropout(dropout)(MME_model)
#########################################

my_matrix_dec_inputs = keras.layers.Input(shape=(my_matrix_maxlen,))


# input_MMD = my_matrix_enc_inputs MMD_model = Embedding(my_matrix_vocab_size+
1,20,embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param)) (my_matrix_dec_inputs)

MMD_model = SpatialDropout1D(0.2)(MMD_model) MMD_strides = return_tuple(protein_strides)
MMD_model = [self.Player(stride_size,filters,activation,initializer,regularizer_param) (MMD_model) for stride_size in MMD_strides]

if len(MMD_model) != 1:

MMD_model = Concatenate(axis=1)(MMD_model) else:
MMD_model = MMD_model[0] MMD_layers = return_tuple(my_matrix_layers)

for my_matrix_layer in MMD_layers:

MMD_model = Dense(256,**params_dict)(MMD_model) MMD_model = BatchNormalization()(MMD_model) MMD_model = Activation(activation)(MMD_model) MMD_model = Dropout(dropout)(MMD_model)
my_matrix_transformer = Transformer(num_layers= my_matrix_num_layers, model_size= my_matrix_model_size, num_heads= my_matrix_num_heads, dff_size= my_matrix_dff_size,
vocab_size= my_matrix_vocab_size+1, maxlen= 256) my_matrix_final_output = my_matrix_transformer([MME_model, MMD_model]) my_matrix_final_output = SpatialDropout1D(0.2)( my_matrix_final_output) my_matrix_final_output = Convolution1D(filters=128,kernel_size=15, padding='same',
kernel_initializer='glorot_normal',

kernel_regularizer=l2(0.001))( my_matrix_final_output) my_matrix_final_output = Activation('relu')( my_matrix_final_output) my_matrix_final_output = GlobalMaxPooling1D()(my_matrix_final_output) #final_output = Dropout(dropout)(final_output)
#final_output = MaxPooling1D()(final_output)

my_matrix_final_output = Dense(64,'relu',**params_dict)( my_matrix_final_output)



#####1111111111111111111111	step1 define protein transformer num_layers = 2
model_size = 20

num_heads = 5

dff_size = 64

maxlen = 800

vocab_size = 474

enc_inputs = keras.layers.Input(shape=(maxlen,)) dec_inputs = keras.layers.Input(shape=(maxlen,))
transformer = Transformer(num_layers=num_layers, model_size=model_size, num_heads=num_heads, dff_size=dff_size,

vocab_size=vocab_size+1, maxlen=maxlen) final_output = transformer([enc_inputs, dec_inputs]) final_output = SpatialDropout1D(0.2)(final_output)
final_output = Convolution1D(filters=128,kernel_size=15, padding='same', kernel_initializer='glorot_normal',

kernel_regularizer=l2(0.001))(final_output) final_output = Activation('relu')(final_output)
final_output = GlobalMaxPooling1D()(final_output) #final_output = Dropout(dropout)(final_output) #final_output = MaxPooling1D()(final_output)
final_output = Dense(64,'relu',**params_dict)(final_output)



###################222222222222222	step 2 define drug morgan-fg input_d = Input(shape=(drug_len,))
drug_layers = return_tuple(drug_layers) for layer_size in drug_layers:
model_d = Dense(layer_size,**params_dict)(input_d) model_d = BatchNormalization()(model_d)
model_d = Activation(activation)(model_d) model_d = Dropout(dropout)(model_d)

##################3333333333333333	step3 define drug convolution input_d2 = Input(shape=(drug_len2,))
model_d2 = Embedding(42, 10, embeddings_initializer=initializer, embeddings_regularizer=l2(regularizer_param))(input_d2)

model_d2 = SpatialDropout1D(0.2)(model_d2) #protein_strides = return_tuple(protein_strides)
model_d2 = [self.Player(10, 128, activation, initializer, regularizer_param)(model_d2) ] if len(model_d2) != 1:
model_d2 = Concatenate(axis=1)(model_d2) else:
model_d2 = model_d2[0]

protein_layers = return_tuple(protein_layers) for protein_layer in protein_layers:
model_d2 = Dense(64, **params_dict)(model_d2) model_d2 = BatchNormalization()(model_d2) model_d2 = Activation(activation)(model_d2) model_d2 = Dropout(dropout)(model_d2)


###################4444444444444444 step 4	define protein convolution input_p = Input(shape=(prot_len,))
model_p = Embedding(vocab_size+ 1,20,embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param))(input_p)

model_p = SpatialDropout1D(0.2)(model_p) protein_strides = return_tuple(protein_strides)

model_p = [self.Player(stride_size,filters,activation,initializer,regularizer_param)(model_p) for stride_size in protein_strides]

if len(model_p) != 1:

model_p = Concatenate(axis=1)(model_p) else:
model_p = model_p[0]

protein_layers = return_tuple(protein_layers) for protein_layer in protein_layers:
model_p = Dense(64,**params_dict)(model_p) model_p = BatchNormalization()(model_p) model_p = Activation(activation)(model_p) model_p = Dropout(dropout)(model_p)
#################55555555555555	step5	concat drug and protein model respectively

finalmodel_D = Concatenate(axis=1)([model_d,model_d2]) finalmodel_D = Dense(64,**params_dict)(finalmodel_D)
# finalmodel_P = Concatenate(axis=1)([model_p,final_output])

finalmodel_P = Concatenate(axis=1)([model_p,final_output,my_matrix_final_output]) finalmodel_P = Dense(64, **params_dict)(finalmodel_P)
model_ttt = Concatenate(axis=1)([finalmodel_D,finalmodel_P]) fc_layers =return_tuple(fc_layers)
for fc_layer in fc_layers:

model_ttt = Dense(units=fc_layer,**params_dict)(model_ttt) model_ttt = Activation(activation)(model_ttt)
#model_ttt = Dense(64,**params_dict)(model_ttt)

model_ttt = Dense(1,activation='sigmoid',activity_regularizer=l2(regularizer_param),
**params_dict)(model_ttt)

model_final = Model(inputs=[input_d,input_d2,input_p,enc_inputs,dec_inputs,my_matrix_enc_inputs,my_ma trix_dec_inputs],outputs=model_ttt)

#plot_model(model_final, to_file='model.png', show_shapes=True, show_layer_names=True) # ä¿ å˜æ¨¡åž‹ç»“æž„å›¾

return model_final

# def  init (self,dropout=0.2,drug_layers=512,protein_strides=15,filters=64,learning_rate= 0.0001,decay=0.0,drug_len2=100,

#	fc_layers=None,prot_vec=None,prot_len=2500,activation='relu',drug_len= 2048,drug_vec='ECFP4',protein_layers=None):

def  init (self, dropout=0.2,drug_layers=512,protein_strides=15,filters=64,learning_rate= 0.0001,decay=0.0,drug_len2=100,

fc_layers=None,prot_vec=None,prot_len=2500,activation='relu',drug_len= 2048,drug_vec='ECFP4',protein_layers=None,

my_matrix_len=21411, my_matrix_layers=None, my_matrix_vec=None):



self.  dropout = dropout

self.  drugs_layer = drug_layers

self. protein_strides = protein_strides self. filters = filters
self. learning_rate = learning_rate self. decay = decay
self. fc_layers = fc_layers self. prot_vec = prot_vec self. prot_len = prot_len

self. activation = activation self. drug_len = drug_len self. drug_vec = drug_vec
self. prot_layers = protein_layers self	drug_len2 = drug_len2
self. my_matrix_len = my_matrix_len self. my_mat_layers = my_matrix_layers self. my_matrix_vec = my_matrix_vec


self.  model_t = self.modelvv(self.  dropout,self.  drugs_layer,self.  protein_strides,self.
  filters,self.  fc_layers,

prot_vec=self.  prot_vec,prot_len=self.  prot_len,activation=self.
  activation,

protein_layers=self.  prot_layers,drug_vec=self.
  drug_vec,drug_len=self.  drug_len,

drug_len2=self.  drug_len2,my_matrix_len = self.  my_matrix_len,

my_matrix_layers = self.  my_mat_layers,my_matrix_vec = self.
  my_matrix_vec)

opt = Adam(lr=learning_rate)

self. model_t.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy']) #tf.compat.v1.keras.backend.get_session(tf.compat.v1.global_variables_initializer())
def summary(self):
self. model_t.summary() # def
validation(self,drug_feature,drug_feature2,protein_feature,protein_feature2,my_matrix,Label, n_epoch=10,batch_size =32,**kwargs):

def

validation(self,drug_feature,drug_feature2,protein_feature,protein_feature2,my_matrix_featur e,Label,n_epoch=10,batch_size =16,**kwargs):

auc_temp = 0 auclist = [] auprlist = []
for i in range(n_epoch):

# self.
  model_t.fit([drug_feature,drug_feature2,protein_feature,protein_feature2,protein_feature2]
,Label,epochs=i+1,verbose=1,initial_epoch=i,batch_size=batch_size,shuffle=True) # with tf.device('/GPU:0'):
self.
 model_t.fit([drug_feature,drug_feature2,protein_feature,protein_feature2,protein_feature2, my_matrix_feature,my_matrix_feature],Label,epochs=i+1,verbose= 1,initial_epoch=i,batch_size=batch_size,shuffle=True)






for dataset in kwargs: print("\tPredction of " + dataset)
##########get the test data feature and use them to predict test_p = kwargs[dataset]["protein_feature"]
test_p2 = kwargs[dataset]["protein_feature2"] test_d = kwargs[dataset]["drug_feature"] test_d2 = kwargs[dataset]['drug_feature2'] test_m = kwargs[dataset]['my_matrix_feature'] test_label = kwargs[dataset]["Label"]

# with tf.device('/GPU:0'):

prediction = self. model_t.predict([test_d,test_d2, test_p,test_p2,test_p2,test_m,test_m])

#####################evaluate our model

fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction) AUC = auc(fpr, tpr)
precision, recall, thresholds_AUPR = precision_recall_curve(test_label, prediction) AUPR = auc(recall, precision)
distance = []

for i in range(len(tpr)): distance.append(tpr[i] - fpr[i])
opt_AUC = thresholds_AUC[distance.index(max(distance))] auclist.append(AUC)
auprlist.append(AUPR) if (AUC > auc_temp):
print('valid AUC : ',AUC,' ******************** test data') auc_temp = AUC
################

testcsv = 'test.csv'

morgantest = 'morgan_test.csv' proteintest = 'protein_test.csv' my_matrix_test = 'my_matrix_test.csv'
finalsets = parse_data(testcsv,morgantest,proteintest,my_matrix_test) print(finalsets)
drug_fea = finalsets['drug_feature']

drug_fea2 = finalsets['drug_feature2'] protein_fea = finalsets['protein_feature'] protein_fea2 = finalsets['protein_feature2']
my_matrix_feature = finalsets['my_matrix_feature'] lab = finalsets['Label']
# with tf.device('/GPU:0'):

final_predition = self. model_t.predict([drug_fea, drug_fea2, protein_fea, protein_fea2, protein_fea2, my_matrix_feature, my_matrix_feature])

finalfpr,finaltpr,finalthresholds = roc_curve(lab,final_predition) finalprecision,finalrecall,finalthres = precision_recall_curve(lab,final_predition) finalAUC = auc(finalfpr,finaltpr)
finalAUPR = auc(finalrecall,finalprecision)



dis = []

for i in range(len(finaltpr)): dis.append(finaltpr[i] - finalfpr[i])
opt = finalthresholds[dis.index(max(dis))] y_preddd = []
for i in range(len(final_predition)):

if final_predition[i] >= opt: y_preddd.append(1)
else:

y_preddd.append(0)

confusion_matix = confusion_matrix(lab, y_preddd)

ACC = (confusion_matix[0][0] + confusion_matix[1][1]) / (

confusion_matix[0][0] + confusion_matix[0][1] + confusion_matix[1][0] + confusion_matix[1][1])
F1 = f1_score(lab, y_preddd)

Sensi = confusion_matix[0][0] / (confusion_matix[0][0] + confusion_matix[0][1]) Speci = confusion_matix[1][1] / (confusion_matix[1][1] + confusion_matix[1][0]) print('threshold_AUC', opt)
print('ACC : ', ACC)

print('AUC : ', finalAUC) print('AUPR : ', finalAUPR) print('Sensitivity : ', Sensi) print('Specificity : ', Speci) print('F1 score : ', F1) out1 = open('1.csv', 'w') out1.write('fpr,tpr\n')
for i in range(len(finalfpr)): out1.write(str(finalfpr[i])) out1.write(',') out1.write(str(finaltpr[i])) out1.write('\n')
out1.close()

#########save AUPR plt to 2.csv out2 = open('2.csv', 'w') out2.write('recall,precision\n') for i in range(len(finalprecision)):

out2.write(str(finalrecall[i])) out2.write(',') out2.write(str(finalprecision[i])) out2.write('\n')
out2.close()



################### #################


y_pred = []

for i in range(len(prediction)): if prediction[i] >= opt_AUC:
y_pred.append(1) else:
y_pred.append(0)

confusion_matix = confusion_matrix(test_label, y_pred) ACC = (confusion_matix[0][0] + confusion_matix[1][1]) / (
confusion_matix[0][0] + confusion_matix[0][1] + confusion_matix[1][0] + confusion_matix[1][1])

Sensi = confusion_matix[0][0] / (confusion_matix[0][0] + confusion_matix[0][1]) Speci = confusion_matix[1][1] / (confusion_matix[1][1] + confusion_matix[1][0]) F1 = f1_score(test_label, y_pred)
print('\n\n\n')

print("\t \t ACC:\t ", ACC)

print('\t optimal threshold(ACC): \t ', opt_AUC)

print("\t \t AUC:\t ", AUC)

print('\t optimal threshold(AUC): \t ', opt_AUC) print("\t \t AUPR:\t ", AUPR)
print('\t optimal threshold(AUPR): \t ', opt_AUC) print("\t \t Sensitivity:\t ", Sensi)
print('\t optimal threshold(Sensitivity): \t ', opt_AUC) print("\t \t Specificity:\t ", Speci)
print('\t optimal threshold(Specificity): \t ', opt_AUC) print("\t \t F1_score:\t ", F1)
print('\t optimal threshold(F1_score): \t ', opt_AUC) print("=================================================")


def save(self, output_file):

self.  model_t.save(output_file)


if  name	== ' main ': import argparse
parser = argparse.ArgumentParser() #train data
parser.add_argument('--dti_dir',default='train.csv') parser.add_argument('--drug_dir',default='morgan_train.csv') parser.add_argument('--protein_dir',default='protein_train.csv')
parser.add_argument('--my_matrix_dir', default= 'my_matrix_train.csv') # New argument for my_matrix_train

#valid data

parser.add_argument('--test-name','-n',default='data') parser.add_argument('--test-dti-dir','-i',default='valid.csv') parser.add_argument('--test-drug-dir','-d',default='morgan_valid.csv') parser.add_argument('--test-protein-dir','-t',default='protein_valid.csv')
parser.add_argument('--test_my_matrix_dir', default= 'my_matrix_valid.csv') # New argument for my_matrix_valid

#struc params

parser.add_argument('--window-sizes','-w',type=int,default=15) parser.add_argument('--protein-layers','-p',type=int,default=64) parser.add_argument('--drug-layers','-c',type=int,default=128) parser.add_argument('--fc-layers','-f',type=int,default=64)
parser.add_argument('--my_matrix_layers', type=int, default=64) # New argument for my_matrix_layers

#training params

parser.add_argument('--learning-rate','-r',default=0.0001,type=float) parser.add_argument('--n-epoch','-e',default=38,type=int)
#type params

parser.add_argument('--prot-vec','-v',default='Convolution') parser.add_argument('--prot-len','-l',default=800,type=int) parser.add_argument('--drug-vec','-V',default='morgan_fp') parser.add_argument('--drug-len','-L',default=2048,type=int) parser.add_argument('--drug-len2',default=100,type=int) parser.add_argument('--my_matrix-vec',default='Convolution')
parser.add_argument('--my_matrix_len', default=21411, type=int) # New argument for my_matrix_len

#other params

parser.add_argument('--activation','-a',default='relu',type=str) parser.add_argument('--dropout','-D',default=0.2,type=float) parser.add_argument('--n-filters','-F',default=128,type=int) parser.add_argument('--batch-size','-b',type=int,default=32) parser.add_argument('--decay','-y',default=0.0001,type=float) #mode params
parser.add_argument('--validation',action='store_true') args = parser.parse_args()
#traindata dic traindata_dic = {
'dti_dir': args.dti_dir, 'drug_dir': args.drug_dir, 'protein_dir': args.protein_dir,
'my_matrix_dir': args.my_matrix_dir, # Load and preprocess my_matrix_train

}

# my_matrix_dic = {

#	'train_my_matrix':args.train_my_matrix # Load and preprocess my_matrix_train # }
#pack the test datasets testnames = args.test_name, validdata_dic = {
'test_dti_dir': args.test_dti_dir, 'test_drugs_dir': args.test_drug_dir,

'test_proteins_dir': args.test_protein_dir, 'test_my_matrix_dir': args.test_my_matrix_dir
}

# test_sets = zip(testnames,test_dti,test_drugs,test_proteins,test_my_matrix)



drug_layers = args.drug_layers protein_layers = args.protein_layers my_matrix_layers = args.my_matrix_layers window_sizes = args.window_sizes fc_layers = args.fc_layers
#training params dict training_params_dict = {
'n_epoch' : args.n_epoch, 'batch_size' : args.batch_size,
}

#type params dict type_params_dict = {
'prot_vec' : args.prot_vec,

'my_matrix_vec' : args.my_matrix_vec, # New argument for my_matrix_vec 'prot_len' : args.prot_len,
'drug_vec' : args.drug_vec, 'drug_len' : args.drug_len, 'drug_len2' : args.drug_len2,
'my_matrix_len' : args.my_matrix_len # New argument for my_matrix_len

}

#Network params network_params = {
'drug_layers' : args.drug_layers, 'protein_strides': args.window_sizes, 'protein_layers' : args.protein_layers, 'fc_layers' : args.fc_layers, 'learning_rate' : args.learning_rate, 'decay' : args.decay,
'activation' : args.activation, 'filters' : args.n_filters, 'dropout' : args.dropout,
'my_matrix_layers' : args.my_matrix_layers # New argument for my_matrix_layers

}

network_params.update(type_params_dict) print('\t model parameters summary \t')
print('=====================================')

for key in network_params.keys():

print('{:20s}: {:10s}'.format(key,str(network_params[key])))



dti_prediction_model = Net(**network_params)



dti_prediction_model.summary()



# with tf.device('/GPU:0'): traindata_dic.update(type_params_dict) traindata_dic = parse_data(**traindata_dic)
# my_matrix = pd.read_csv("my_matrix_train.csv").values # traindata_dic['my_matrix'] = my_matrix


# test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, test_my_matrix,
**type_params_dict)

#	for test_name, test_dti, test_drug, test_protein, test_my_matrix in test_sets} # print(f"test_dic{test_dic}")
# my_matrix_valid = pd.read_csv("my_matrix_valid.csv").values # test_dic['my_matrix'] = my_matrix_valid
validdata_dic = parse_data(validdata_dic['test_dti_dir'],validdata_dic['test_drugs_dir'],validdata_dic['test_prot eins_dir'],validdata_dic['test_my_matrix_dir'],**type_params_dict)

##########validation params validation_params = {}
# with tf.device('/GPU:0'): validation_params.update(training_params_dict) print("\tvalidation summary\t")
print("=====================================================")

for key in validation_params.keys():

print("{:20s} : {:10s}".format(key, str(validation_params[key])))

print("=====================================================")

# with tf.device('/GPU:0'): validation_params.update(traindata_dic) validation_params.update(validdata_dic) print(validation_params)
# with tf.device('/GPU:0'): dti_prediction_model.validation(**validation_params)