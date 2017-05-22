import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import sklearn.preprocessing as pp
import sklearn.linear_model as linear_model
from sklearn.ensemble import RandomForestClassifier

#reading a file

ft = pd.read_csv('C:/Users/sunny/OneDrive/Documents/train_indessa.csv',skip_blank_lines=True);
ft_minus_unrelfeat=ft.drop(['loan_status','verification_status_joint','member_id','funded_amnt' ,'funded_amnt_inv','batch_enrolled', 'grade' ,'sub_grade','emp_title','desc','zip_code', 'recoveries','collection_recovery_fee','collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay','total_acc'], axis=1)     
ft_minus_unrel_feat_labl=ft_minus_unrelfeat.drop(['term','title','home_ownership','verification_status','pymnt_plan','purpose','addr_state','initial_list_status','application_type'],axis=1)
ft_minus_unrel_feat_labl=ft_minus_unrel_feat_labl.fillna(0);
scalar = pp.MinMaxScaler()
#splitting label and continuous values column
file_test_labeled_column = ft[['term','home_ownership','verification_status','pymnt_plan','purpose','addr_state','initial_list_status','application_type']]
file_test_labeled_column = file_test_labeled_column.apply(pp.LabelEncoder().fit_transform)
frame1 = [ft_minus_unrel_feat_labl,file_test_labeled_column]
X_train=pd.concat(frame1,axis=1)
X_train['emp_length'] = X_train['emp_length'].str.extract('(\d+)')

X_train['emp_length'] = X_train['emp_length'].fillna(0)
X_train = scalar.fit_transform(X_train)
X_train=np.c_[np.ones((X_train.shape[0])),X_train]
y_train =ft['loan_status']
#print(X_train.shape)
###########################################LR###############
#logreg = linear_model.LogisticRegression()
#logreg.fit(X_train,y_train)  
#################################LR#########################################
###################################NN########################################
#def baseline_model():
#    # create model
#    model = Sequential()
#    model.add(Dense(8, input_dim=28, init='normal', activation='sigmoid'))
#    model.add(Dense(2, init='normal', activation='sigmoid'))
#    # Compile model
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    return model
#estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=2000, batch_size=100000, verbose=1)
#estimator.fit(X_train, y_train)
#predictions = estimator.pred_prob(X_test)
##################################NN#####################################

######################################RandomForest########################################

clf = RandomForestClassifier(n_estimators=99,verbose=1)
clf.fit(X_train, y_train)
ft = pd.read_csv('C:/Users/sunny/OneDrive/Documents/test_indessa.csv',skip_blank_lines=True);
ft_minus_unrelfeat=ft.drop(['verification_status_joint','member_id','funded_amnt' ,'funded_amnt_inv','batch_enrolled', 'grade' ,'sub_grade','emp_title','desc','zip_code', 'recoveries','collection_recovery_fee','collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay','total_acc'], axis=1)     
ft_minus_unrel_feat_labl=ft_minus_unrelfeat.drop(['term','title','home_ownership','verification_status','pymnt_plan','purpose','addr_state','initial_list_status','application_type'],axis=1)
ft_minus_unrel_feat_labl=ft_minus_unrel_feat_labl.fillna(0);
#splitting label and continuous values column
file_test_labeled_column = ft[['term','home_ownership','verification_status','pymnt_plan','purpose','addr_state','initial_list_status','application_type']]
file_test_labeled_column = file_test_labeled_column.apply(pp.LabelEncoder().fit_transform)
frame = [ft_minus_unrel_feat_labl,file_test_labeled_column]
X_val=pd.concat(frame,axis=1)
X_val['emp_length'] = X_val['emp_length'].str.extract('(\d+)')
X_val['emp_length'] = X_val['emp_length'].fillna(0)
X_val=np.c_[np.ones((X_val.shape[0])),X_val]
df = ft['member_id']
#z=logreg.predict(X_val)
z = clf.predict_proba(X_val)
df = np.array(df)
df = np.c_[df,z[:,1]]
np.savetxt("C:/Users/sunny/OneDrive/Documents/final.csv",df, delimiter=",")
