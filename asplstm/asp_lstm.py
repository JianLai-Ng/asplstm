#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import datetime as dt
from copy import deepcopy
from numpy import array
import sys
import csv
from csv import writer
import json
import argparse
import matplotlib.pyplot as plt
import math
import time
from dateutil import parser
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from copy import deepcopy





pd.set_option('display.max_columns', None)


from numpy.random import seed
seed(123)

try:
    tf.set_random_seed(123)
except:
    tf.random.set_seed(123)




# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X=[]
    y = []
    for i in range(len(sequences)):
        #print(i)
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix , :], sequences[end_ix:out_end_ix , :]
        X.append(seq_x)
        y.append(seq_y)
    #print(X)
    #print(y)
    return array(X), array(y)

#appends entry to csv with column and overwrites
def append_list_as_row(nama, list_of_elem):
    # Open file in append mode
        with open(nama, 'a+',encoding='utf8', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
                

class PSA_LSTM:
    def __init__(self, n_steps_in, n_steps_out,y_vars,granularity,not_Trial):
        print("Initialising")
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.y_scale = None
        self.validation_date = '2019-07-01'
        self.test_date= '2019-10-01'
        self.y_vars = y_vars
        self.granularity = granularity
        self.data = None
        self.not_Trial = not_Trial
        
        self.X = None
        self.y = None
        self.vali_X = None 
        self.vali_y = None
        self.test_X = None
        self.test_y = None
        
        self.saveFileName = None
        
        self.load_data()
        self.data_preproc()

        
    def load_data(self):
        if ('BOX_TEU' == self.y_vars) and ('day' == self.granularity):
            self.data = pd.read_csv("LSTM_TEU_BOX_days_scaled.csv")
            self.y_scale = {'SUMOFBOX_Q': 42307.0, 'SUMOFTEU_Q': 66543.875}
            print("Data successfully loaded")
            print("")
        else:
            print(self.y_vars)
            print((self.y_vars == 'BOX_TEU'))
            print(self.granularity)
            print((self.granularity == 'day'))
            print("Load data unsuccessful")
            
        
        

    


    def data_preproc(self):
        usable = self.data
        self.data.index = self.data['ATD1_day']
        
        #Test set
        test_input_output = usable[(usable.index >= self.test_date )]
        test_input_output.reset_index(drop=True, inplace=True)
        test_input_output = test_input_output.drop(["ATD1_day"], axis=1)
        #print(test_input_output.shape)
        
        #Validation set
        vali_input_output = usable[(usable.index < self.test_date ) &(usable.index >= self.validation_date ) ]
        vali_input_output.reset_index(drop=True, inplace=True)
        vali_input_output = vali_input_output.drop(["ATD1_day"], axis=1)
        print(vali_input_output.shape)
        
        #Train set
        train_input_output = usable[(usable.index < self.validation_date)]
        train_input_output.reset_index(drop=True, inplace=True)
        train_input_output = train_input_output.drop(["ATD1_day"], axis=1)
        print(train_input_output.shape)


        #prepare train_data
        exc = train_input_output.drop(['SUMOFBOX_Q','SUMOFTEU_Q'],axis=1)
        num_feat = exc.shape[1]
        wai = train_input_output[['SUMOFBOX_Q','SUMOFTEU_Q']]
        #shift to back
        exc[['SUMOFBOX_Q','SUMOFTEU_Q']]= wai
        
        train_index = exc.index
        dataset = exc.to_numpy()
        y_adjust = train_input_output.shape[1]-2
        # convert into input/output
        self.X, self.y = split_sequences(dataset, self.n_steps_in, self.n_steps_out)
        ##Seperate features only in X and lag y and result only in y
        self.y = self.y[:,:,y_adjust:]
        # the dataset knows the number of features, e.g. 2
        n_features = self.X.shape[2]
        print("Train X.shape: "+ str(self.X.shape))
        print("Train y.shape: "+ str(self.y.shape))
        
        #prepare validation data
        exc = vali_input_output.drop(['SUMOFBOX_Q','SUMOFTEU_Q'],axis=1)
        num_feat = exc.shape[1]
        #exc = exc.drop(['date'],axis=1)
        wai = vali_input_output[['SUMOFBOX_Q','SUMOFTEU_Q']]
        #shift to back
        exc[['SUMOFBOX_Q','SUMOFTEU_Q']]= wai
        vali_index = exc.index
        dataset = exc.to_numpy()
        y_adjust = vali_input_output.shape[1]-2
        # choose a number of time steps
        # covert into input/output
        self.vali_X, self.vali_y = split_sequences(dataset, self.n_steps_in, n_steps_out)
        ##Seperate features only in X and lag y and result only in y
        self.vali_y = self.vali_y[:,:,y_adjust:]
        # the dataset knows the number of features, e.g. 2
        print("Validation X.shape: "+ str(self.vali_X.shape))
        print("Validation y.shape: "+ str(self.vali_y.shape))
        
        #prepare test data       
        exc = test_input_output.drop(['SUMOFBOX_Q','SUMOFTEU_Q'],axis=1)
        num_feat = exc.shape[1]
        #exc = exc.drop(['date'],axis=1)
        wai = test_input_output[['SUMOFBOX_Q','SUMOFTEU_Q']]
        #shift to back
        exc[['SUMOFBOX_Q','SUMOFTEU_Q']]= wai
        test_index = exc.index
        dataset = exc.to_numpy()
        y_adjust = test_input_output.shape[1]-2
        # choose a number of time steps
        #n_steps_in, n_steps_out = 14, 14
        # covert into input/output
        self.test_X, self.test_y = split_sequences(dataset, n_steps_in, n_steps_out)
        ##Seperate features only in X and lag y and result only in y
        self.test_y = self.test_y[:,:,y_adjust:]
        # the dataset knows the number of features, e.g. 2
        print("Test X.shape: "+ str(self.test_X.shape))
        print("Test y.shape: "+ str(self.test_y.shape))
        print("End of data splitting & shaping")
        print("")
    

    def grid_and_model(self):
        

        if self.not_Trial == True: #actual run
            def scheduler0(epoch, lr):
                return lr

            def scheduler1(epoch, lr):
              if epoch < 50:
                return lr
              else:
                return lr * tf.math.exp(-0.1)

            def scheduler2(epoch, lr):
              if epoch < 50:
                return lr
              else:
                return lr * tf.math.exp(-0.05)

            def scheduler3(epoch, lr):
              if epoch < 50:
                return lr
              else:
                return lr * tf.math.exp(-0.01)

            #ACTUAL RUN
            neurons_L = [500,300,200]
            activation_L = ['relu','tanh']
            optimiser_L = ['RMSprop','Adagrad','SGD']
            epoch_L = [50, 100,150]
            learning_rate_L = ['None','exp-0.1','exp-0.05' ,'exp-0.01' ]


            optim_d={}
            optim_d['RMSprop']=tf.keras.optimizers.RMSprop()
            optim_d['Adagrad']=tf.keras.optimizers.Adagrad()
            optim_d['SGD']=tf.keras.optimizers.SGD()


            learnR_d={}
            learnR_d['None']=tf.keras.callbacks.LearningRateScheduler(scheduler0)
            learnR_d['exp-0.1']=tf.keras.callbacks.LearningRateScheduler(scheduler1)
            learnR_d['exp-0.05']=tf.keras.callbacks.LearningRateScheduler(scheduler2)
            learnR_d['exp-0.01']=tf.keras.callbacks.LearningRateScheduler(scheduler3)
            print("Running Actual Sequence")
            print("")

        else: #trial run
            
            def scheduler0(epoch, lr):
                return lr
            optim_d={}
            optim_d['RMSprop']=tf.keras.optimizers.RMSprop()
            
            learnR_d={}
            learnR_d['None']=tf.keras.callbacks.LearningRateScheduler(scheduler0)
            
            neurons_L = [10,5]
            activation_L = ['relu']
            optimiser_L = ['RMSprop']
            epoch_L = [2]
            learning_rate_L = ['None']
            print("Running Trial Sequence")
            print("")

        modelNum = 0
        total_model_numbers= len(neurons_L)*len(activation_L)*len(optimiser_L)*len(epoch_L)*len(learning_rate_L)
        print("Total number of models to run: " + str(total_model_numbers))
        print("")
    
        if ('BOX_TEU' in self.y_vars)  and ('day' in self.granularity):
            columns =['modelNumber', 'neurons', 'activation', 'optimiser', 'epoch', 'learning_rate',
                                              'train_MSE_box','train_MAE_box','train_MSE_teu','train_MAE_teu',
                                             'validation_MSE_box' ,'validation_MAE_box','validation_MSE_teu','validation_MAE_teu',
                                             'test_MSE_box' ,'test_MAE_box','test_MSE_teu','test_MAE_teu']
        else:
            print("No intial CSV created")
            
        with open(r'Overview_'+ str(self.n_steps_in)+'_' +str(self.n_steps_out)+'_'+self.granularity+'_'+self.y_vars +'.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

        self.saveFileName = 'Overview_'+ str(self.n_steps_in)+'_' +str(self.n_steps_out)+'_'+self.granularity+'_'+self.y_vars +'.csv'
#######
        #START RUNNING
        Best_BOX_MSE = 99999999999999999999999
        Best_BOX_MAE = 99999999999999999999999
        Best_TEU_MSE = 99999999999999999999999
        Best_TEU_MAE = 99999999999999999999999
        
        best_models_CAT_NAME ={}
        best_models_CAT_MODEL = {}
    
        for neu in neurons_L:
            for act in activation_L:
                for optKEY in optimiser_L:
                    for ep in epoch_L:
                        for reduce_lrKEY in learning_rate_L:
                           # try:
                            to_save = False
                            
                            opt = optim_d[optKEY]
                            reduce_lr = learnR_d[reduce_lrKEY]

                              #   E2D2   # Sequence to Sequence Model with two encoder layers and two decoder layers.
                    ########################################## BUILD MODEL ##############################################
                            modelNum+=1
                            print(modelNum)
                            model_code = str(modelNum) + "_"+ str(neu) +"Neus_" + str(act) + "_"+ str(optKEY)+ "_"+str(ep)+"Ep_" + str(reduce_lrKEY)+ "LR"+ "_"
                            total_model_numbers -=1

                            print("Training Model " + str(modelNum))
                            print("Remaining:  " + str(total_model_numbers))

                            encoder_inputs = tf.keras.layers.Input(shape=(n_steps_in, self.X.shape[2]))
                            encoder_l1 = tf.keras.layers.LSTM(neu, activation=act, return_sequences = True, return_state=True)
                            encoder_outputs1 = encoder_l1(encoder_inputs)
                            encoder_states1 = encoder_outputs1[1:]
                            encoder_l2 = tf.keras.layers.LSTM(neu, activation=act, return_state=True)
                            encoder_outputs2 = encoder_l2(encoder_outputs1[0])
                            encoder_states2 = encoder_outputs2[1:]
                            #
                            decoder_inputs = tf.keras.layers.RepeatVector(self.n_steps_out)(encoder_outputs2[0])
                            #
                            decoder_l1 = tf.keras.layers.LSTM(neu, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
                            decoder_l2 = tf.keras.layers.LSTM(neu, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
                            decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.y.shape[2]))(decoder_l2)
                            #
                            model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)

                            model_e2d2.compile(optimizer=opt, loss=tf.keras.losses.Huber())
                            
                            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                            
                            history_e2d2=model_e2d2.fit(self.X,self.y,epochs=ep,validation_data=(self.vali_X,self.vali_y),batch_size=32,verbose=1,callbacks=[reduce_lr,early_stop],shuffle=False)




                    ################################# TRAIN & VALIDATION LOSS PLOT #######################################
                            history = history_e2d2
                            model = model_e2d2
                            
                            ep = len(history.history['loss'])

                            fig, ax1 = plt.subplots(figsize=(25, 9))
                            plt.plot(history.history['loss'])
                            plt.plot(history.history['val_loss'])
                            plt.title('Loss_'+ str(model_code ))
                            plt.ylabel('Loss')
                            plt.xlabel('Epoch')
                            plt.legend(['training', 'validation'], loc='upper right')
                            fname_modelLoss = 'model_E2D2_' + str(model_code ) +'Loss'
                            plt.savefig(fname = fname_modelLoss)
                            #plt.show()



                    ################################################# PREDICT ############################################
                            ######################TRAINING DATA

                            boxTrain_yhats = []
                            teuTrain_yhats = []

                            boxTrain_ys = []
                            teuTrain_ys = []

                            #predict all training instances
                            for train_instance_in in range(len(self.X)):
                                train_X_in = self.X[train_instance_in]
                                train_y_in = self.y[train_instance_in]

                            #PRED
                                x_input = train_X_in.reshape((1, self.n_steps_in, self.X.shape[2]))
                                train_yhat = model.predict(x_input, verbose=0)
                                box_pred = [x[0] for x in train_yhat[0]]
                                teu_pred = [x[1] for x in train_yhat[0]]

                                #rescale predictions column to original
                                box_pred_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_pred]
                                teu_pred_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_pred]

                                #add to a long list
                                boxTrain_yhats+=box_pred_rescaled
                                teuTrain_yhats+=teu_pred_rescaled

                            #ACTUAL
                                box_actual = [x[0] for x in train_y_in]
                                teu_actual = [x[1] for x in train_y_in]

                                #rescale actual column to original
                                box_actual_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_actual]
                                teu_actual_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_actual]

                                #add to a long list
                                boxTrain_ys+=box_actual_rescaled
                                teuTrain_ys+=teu_actual_rescaled
                                #rescale actual numbers



                            #Train BOX
                            train_MSE_box = mean_squared_error(boxTrain_yhats, boxTrain_ys)
                            train_MAE_box = mean_absolute_error(boxTrain_yhats, boxTrain_ys)



                            #Train TEU
                            train_MSE_teu = mean_squared_error(teuTrain_yhats, teuTrain_ys)
                            train_MAE_teu = mean_absolute_error(teuTrain_yhats, teuTrain_ys)


                            ######################Validation DATA


                            boxValidation_yhats = []
                            teuValidation_yhats = []

                            boxValidation_ys = []
                            teuValidation_ys = []

                            #predict all Validationing instances
                            for Validation_instance_in in range(len(self.vali_X)):
                                Validation_X_in = self.vali_X[Validation_instance_in]
                                Validation_y_in = self.vali_y[Validation_instance_in]

                            #PRED
                                x_input = Validation_X_in.reshape((1, self.n_steps_in, self.X.shape[2]))
                                Validation_yhat = model.predict(x_input, verbose=0)
                                box_pred = [x[0] for x in Validation_yhat[0]]
                                teu_pred = [x[1] for x in Validation_yhat[0]]

                                #rescale predictions column to original
                                box_pred_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_pred]
                                teu_pred_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_pred]

                                #add to a long list
                                boxValidation_yhats+=box_pred_rescaled
                                teuValidation_yhats+=teu_pred_rescaled

                            #ACTUAL
                                box_actual = [x[0] for x in Validation_y_in]
                                teu_actual = [x[1] for x in Validation_y_in]

                                #rescale actual column to original
                                box_actual_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_actual]
                                teu_actual_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_actual]

                                #add to a long list
                                boxValidation_ys+=box_actual_rescaled
                                teuValidation_ys+=teu_actual_rescaled
                                #rescale actual numbers



                            #Validation BOX
                            validation_MSE_box = mean_squared_error(boxValidation_yhats, boxValidation_ys)
                            validation_MAE_box = mean_absolute_error(boxValidation_yhats, boxValidation_ys)



                            #Validation TEU
                            validation_MSE_teu = mean_squared_error(teuValidation_yhats, teuValidation_ys)
                            validation_MAE_teu = mean_absolute_error(teuValidation_yhats, teuValidation_ys)


                            ######################TEST DATA


                            boxtest_yhats = []
                            teutest_yhats = []

                            boxtest_ys = []
                            teutest_ys = []

                            #predict all testing instances
                            for test_instance_in in range(len(self.test_X)):
                                test_X_in = self.test_X[test_instance_in]
                                test_y_in = self.test_y[test_instance_in]

                            #PRED
                                x_input = test_X_in.reshape((1, self.n_steps_in, self.X.shape[2]))
                                test_yhat = model.predict(x_input, verbose=0)
                                box_pred = [x[0] for x in test_yhat[0]]
                                teu_pred = [x[1] for x in test_yhat[0]]

                                #rescale predictions column to original
                                box_pred_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_pred]
                                teu_pred_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_pred]

                                #add to a long list
                                boxtest_yhats+=box_pred_rescaled
                                teutest_yhats+=teu_pred_rescaled

                            #ACTUAL
                                box_actual = [x[0] for x in test_y_in]
                                teu_actual = [x[1] for x in test_y_in]

                                #rescale actual column to original
                                box_actual_rescaled = [element * self.y_scale['SUMOFBOX_Q'] for element in box_actual]
                                teu_actual_rescaled = [element * self.y_scale['SUMOFTEU_Q'] for element in teu_actual]

                                #add to a long list
                                boxtest_ys+=box_actual_rescaled
                                teutest_ys+=teu_actual_rescaled
                                #rescale actual numbers



                            #test BOX
                            test_MSE_box = mean_squared_error(boxtest_yhats, boxtest_ys)
                            test_MAE_box = mean_absolute_error(boxtest_yhats, boxtest_ys)



                            #test TEU
                            test_MSE_teu = mean_squared_error(teutest_yhats, teutest_ys)
                            test_MAE_teu = mean_absolute_error(teutest_yhats, teutest_ys)
                            
                    ################################################# SAVE MODEL #########################################
                    #save model if better

                    
                            if Best_BOX_MSE > validation_MSE_box:
                                Best_BOX_MSE = validation_MSE_box
                                best_models_CAT_NAME['Best_BOX_MSE'] = model_code
                                best_models_CAT_MODEL['Best_BOX_MSE'] = model

    
                            if Best_BOX_MAE > validation_MAE_box:
                                Best_BOX_MAE = validation_MAE_box
                                best_models_CAT_NAME['Best_BOX_MAE'] = model_code
                                best_models_CAT_MODEL['Best_BOX_MAE'] = model
 
                            if Best_TEU_MSE > validation_MSE_teu:
                               Best_TEU_MSE = validation_MSE_teu
                               best_models_CAT_NAME['Best_TEU_MSE'] = model_code
                               best_models_CAT_MODEL['Best_TEU_MSE'] = model
   
                            if Best_TEU_MAE > validation_MAE_teu:
                               Best_TEU_MAE = validation_MAE_teu
                               best_models_CAT_NAME['Best_TEU_MAE'] = model_code
                               best_models_CAT_MODEL['Best_TEU_MAE'] = model

                              
                            
                            fname_modelFile = 'model_E2D2_' + str(model_code ) +'.h5'
                            model.save(fname_modelFile)










        ################################################# Plot last prediction #################################################



                            box_prediction = pd.DataFrame({'Predicted_box':box_pred_rescaled, 'Actual_box':box_actual_rescaled}, index = self.data.index[-(self.n_steps_out):])
                            teu_prediction = pd.DataFrame({'Predicted_teu':teu_pred_rescaled, 'Actual_teu':teu_actual_rescaled}, index = self.data.index[-(self.n_steps_out):])

                            #BOX

                            fig, ax1 = plt.subplots(figsize=(25, 9))

                            toPlot = self.data[self.data.index>= '2019-01-01']

                            plt.plot(toPlot.index, toPlot['SUMOFBOX_Q']*self.y_scale['SUMOFBOX_Q'], label="Actual SUMOFBOX_Q")
                            plt.plot(box_prediction.index, box_prediction['Predicted_box'], label="Predicted SUMOFBOX_Q")

                            plt.xlabel('Dates')
                            plt.ylabel('Units')

                            plt.title('SUMOFBOX_Q')

                            plt.legend()

                            plt.xticks(['2019-01-01','2019-06-01','2019-12-01'])
                            plt.xticks(fontsize=5, rotation=10, horizontalalignment='center')
                            ax1.grid(True)
                            plt.tick_params(labelsize=18)

                            fname_modelperf = 'model_E2D2_' + str(model_code) +'Box_TEST_Performance'
                            plt.savefig(fname = fname_modelperf)
                            #plt.show()

                            #TEU

                            fig, ax1 = plt.subplots(figsize=(25, 9))

                            toPlot = self.data[self.data.index>= '2019-01-01']

                            plt.plot(toPlot.index, toPlot['SUMOFTEU_Q']*self.y_scale['SUMOFTEU_Q'], label="Actual SUMOFTEU_Q")
                            plt.plot(teu_prediction.index, teu_prediction['Predicted_teu'], label="Predicted SUMOFTEU_Q")

                            plt.xlabel('Dates')
                            plt.ylabel('Units')

                            plt.title('SUMOFTEU_Q')

                            plt.legend()

                            plt.xticks(['2019-01-01','2019-06-01','2019-12-01'])
                            plt.xticks(fontsize=5, rotation=10, horizontalalignment='center')
                            ax1.grid(True)
                            plt.tick_params(labelsize=18)

                            fname_modelperf = 'model_E2D2_' + str(model_code) +'TEU_TEST_Performance'
                            plt.savefig(fname = fname_modelperf)
                            #plt.show()



        ################################################# Update results #################################################


                            if ('BOX_TEU' in self.y_vars)  and ('day' in self.granularity):
                                entry =[modelNum, neu, act, optKEY, ep, reduce_lrKEY, train_MSE_box,train_MAE_box,train_MSE_teu,train_MAE_teu, validation_MSE_box ,validation_MAE_box,validation_MSE_teu,validation_MAE_teu,test_MSE_box ,test_MAE_box,test_MSE_teu,test_MAE_teu]

                                append_list_as_row(self.saveFileName, entry)
                            else:
                                print("Result not appended")

#                             except:
#                                 err =[modelNum, neu, act, optKEY, ep, reduce_lrKEY]
#                                 print("ERROR Running: "+ str(err))    


        ######### SAVE BEST MODELS ##############


        for model_key in best_models_CAT_MODEL:
            this_model = best_models_CAT_MODEL[model_key]

            fname_modelFile = 'model_E2D2_' + str(best_models_CAT_NAME[model_key]) +model_key+'.h5'
            this_model.save(fname_modelFile)

        with open('best_models.json', 'w') as fp:
            json.dump(best_models_CAT_NAME, fp)



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inS', type= int, default=14,
                         help='steps_in')
    
    parser.add_argument('--outS', type=int, default=14,
                         help='out')
    
    parser.add_argument('--y', type=str, default='BOX_TEU',
                         help='y')
    
    parser.add_argument('--gran', type=str, default='day',
                         help='granularity')
    
    parser.add_argument('--notTrial', type=int, default = 1,
                         help='trial_run?')
    

    args = parser.parse_args()
    print(args)
    
    n_steps_in =  args.inS
    n_steps_out =  args.outS
    
    y_vars = args.y
    granularity = args.gran
    
    if args.notTrial == 0:
        not_Trial = False
    else:
        not_Trial = True
    

    
    print("INPUTS: ")
    print("  " + str(n_steps_in))
    print("  " + str(n_steps_out))
    print("  " + str(y_vars))
    print("  " + str(granularity))
    print("  " + str(not_Trial))
    
    psa_lstm = PSA_LSTM(int(n_steps_in), int(n_steps_out),y_vars,granularity,not_Trial )

    psa_lstm.grid_and_model()
    