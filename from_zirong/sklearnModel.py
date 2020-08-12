import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class machineLearning:
    __typeLearning = ''
    __labelName = ''

    __model = None
    __dfData =  pd.DataFrame()      
    __xData  = pd.DataFrame()
    __yData  = pd.DataFrame()
    __xTrain = pd.DataFrame() 
    __xTest  = pd.DataFrame() 
    __yTrain = pd.DataFrame() 
    __yTest  = pd.DataFrame() 

    def __init__(self,typeLearning,dataFilePath,labelName,strategyPara,Binarizer=None):
        """---Initialize settings--------------------------------
        The missing data will be filled with mean
        ---Parameters
        typeLearning : Classification or Regression
        dataFilePath : csv file path
        labelName    : Tags that need classification or regression
        strategyPara : How to fill in missing values 
                       ['mean','median','most_frequent']
        ---Return
        None
        ------------------------------------------------------"""
        self.__typeLearning = typeLearning
        self.__labelName = labelName
        self.__dfData = pd.read_excel(io = dataFilePath)
        self.__yData =  self.__dfData[[labelName]]
        self.__dfData.drop(labelName,axis=1,inplace=True)
        self.__xData = self.__dfData.values

        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy=strategyPara)
        self.__xData = imp.fit_transform(self.__xData)

        if Binarizer != None:
            self.__labelBinarizer(Binarizer)
        else:
            self.__yData = self.__yData.values
        self.__yData = self.__yData.flatten()

    def __labelBinarizer(self,threshold):
        """---labelBinarizer-----------------------------------------
        Values greater than the threshold map to 1, 
        while values less than or equal to the threshold map to 0. 
        ---Parameters
        threshold : float
         ---Return
        None
        ------------------------------------------------------"""
        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold=threshold)
        self.__yData = binarizer.fit_transform(self.__yData)
        
    def trainTestSplit(self,testSize,randomState=None):
        """---Split Data-----------------------------------------
        ---Parameters
        testSize : test data size % (0-1) Undesirable 0 and 1
        randomState : default value None
         ---Return
        None
        ------------------------------------------------------"""
        from sklearn.model_selection import train_test_split
        self.__xTrain,self.__xTest, self.__yTrain, self.__yTest = train_test_split(self.__xData, self.__yData,test_size=testSize,random_state = randomState)
    
    def labelMinMaxScaler(self):
        from sklearn.preprocessing import MinMaxScaler
        minMax = MinMaxScaler()
        self.__yData = np.expand_dims(self.__yData,axis=1)
        self.__yData = minMax.fit_transform(self.__yData)
        self.__yData = self.__yData.flatten()

    def dataPretreatment(self,scaler,parameter=None):
        """---data Pretreatment----------------------------------
        ---Parameters
        scaler : [StandardScaler,MinMaxScaler,Normalizer,MaxAbsScaler]
        parameter : if scaler is Normalizer: norm['l1','l2']
         ---Return
        StandardScaler : mean and var
        MinMaxScaler : data_max and data_min
        Normalizer : [],[]
        ------------------------------------------------------"""

        if scaler =='StandardScaler':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.__xData = scaler.fit_transform(self.__xData)
            return scaler.mean_,scaler.var_

        elif scaler == 'MinMaxScaler':
            from sklearn.preprocessing import MinMaxScaler
            minMax = MinMaxScaler()
            self.__xData = minMax.fit_transform(self.__xData)
            return minMax.data_max_,minMax.data_min_

        elif scaler == 'Normalizer':
            from sklearn.preprocessing import Normalizer
            normaLizer = Normalizer(norm= parameter['norm'])
            self.__xData = normaLizer.fit_transform(self.__xData)
            return [],[]

        elif scaler == 'MaxAbsScaler':
            from sklearn.preprocessing import MaxAbsScaler
            maxAbsScaler = MaxAbsScalerMaxAbsScaler()
            self.__xData = maxAbsScaler.fit_transform(self.__xData)
            return [],[]

    def featureSelection(self,Selection):
        """---featureSelection-----------------------------------
        VarianceThreshold : Remove variance to 0
        chi2 : Choose the most useful column
        ---Parameters
        Selection : VarianceThreshold,chi2
        Threshold : Select the number of features
         ---Return
        None
        ------------------------------------------------------"""
        if Selection == 'VarianceThreshold':
            from sklearn.feature_selection import VarianceThreshold
            variance = VarianceThreshold()
            self.__xData = variance.fit_transform(self.__xData)    
            return variance.variances_

        elif Selection == 'chi2':
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import chi2
            selectKBest = SelectKBest(chi2)
            self.__xData = selectKBest.fit_transform(self.__xData, self.__yData)
            return selectKBest.pvalues_
            
    def featureDimensionalityReduction(self,Reduction,nComponents):
        """---featureDimensionalityReduction-----------------------------------
        ---Parameters
        Reduction : PCA,LDA
        nComponents : if PCA nComponents is Select the number of features
                      if LDA None. nComponents default Equal to the number of tags
         ---Return
        None
        ------------------------------------------------------"""
        if Reduction == 'PCA':
            from sklearn.decomposition import PCA           #加载PCA算法包
            pca = PCA(n_components = nComponents)
            self.__xData = pca.fit_transform(self.__xData)

            #cumsum = np.cumsum(pca.explained_variance_ratio_)
            plt.figure()   
            plt.plot(pca.explained_variance_ratio_,linewidth=2)  
            plt.show()                
        elif Reduction == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis()
            self.__xData = lda.fit_transform(self.__xData, self.__yData)

    def __regressionModel(self,model,parameter):
        if model == 'DecisionTree':
            from sklearn import tree
            self.__model = tree.DecisionTreeRegressor()
        elif model == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            self.__model = LinearRegression()
        elif model == 'SVM':
            from sklearn import svm
            self.__model = svm.SVR(kernel=parameter["kernel"],C=parameter["C"])
        elif model == 'KNeighbors':
            from sklearn import neighbors
            self.__model = neighbors.KNeighborsRegressor(n_neighbors=parameter["n_neighbors"])
        elif model == 'RandomForest':
            from sklearn import ensemble
            self.__model = ensemble.RandomForestRegressor(n_estimators=parameter["n_estimators"])
        elif model == 'AdaBoost':
            from sklearn import ensemble
            self.__model= ensemble.AdaBoostRegressor(n_estimators=parameter["n_estimators"])
        elif model == 'GradientBoosting':
            from sklearn import ensemble
            self.__model= ensemble.GradientBoostingRegressor(n_estimators=parameter["n_estimators"])  
        elif model == 'Bagging':
            from sklearn import ensemble
            self.__model = ensemble.BaggingRegressor(n_estimators=parameter["n_estimators"])
        elif model == 'ExtraTree':
            from sklearn.tree import ExtraTreeRegressor
            self.__model = ExtraTreeRegressor()
    
    def __classifierModel(self,model,parameter):
        if model == 'KNN':   
            from sklearn.neighbors import KNeighborsClassifier
            self.__model = KNeighborsClassifier(n_neighbors = parameter["n_neighbors"])
        elif model == 'Logistic':
            from sklearn.linear_model import LogisticRegression
            self.__model = LogisticRegression(penalty=parameter["penalty"])
        elif model == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            self.__model = RandomForestClassifier(n_estimators=parameter["n_estimators"])
        elif model == 'DecisionTree':
            from sklearn import tree
            self.__model = tree.DecisionTreeClassifier()
        elif model == 'GBDT':
            from sklearn.ensemble import GradientBoostingClassifier
            self.__model = GradientBoostingClassifier(n_estimators=parameter["n_estimators"])
        elif model == 'AdaBoost':
            from sklearn.ensemble import  AdaBoostClassifier
            self.__model = AdaBoostClassifier(n_estimators = parameter["n_estimators"])
        elif model == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            self.__model = GaussianNB()
        elif model == 'LinearDiscriminantAnalysis':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.__model = LinearDiscriminantAnalysis()
        elif model == 'QuadraticDiscriminantAnalysis':
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            self.__model = QuadraticDiscriminantAnalysis()
        elif model == 'SVM':
            from sklearn.svm import SVC
            self.__model = SVC(kernel=parameter["kernel"],C=parameter["C"])
        elif model == 'MultinomialNaiveBayes':
            from sklearn.naive_bayes import MultinomialNB
            self.__model = MultinomialNB(alpha=parameter["alpha"])

    def trainingModel(self,model,parameter):
        """---trainingModel-----------------------------------
        ---Parameters
        model : Classification {KNN,Logistic,RandomForest,
                                DecisionTree,
                                GBDT,AdaBoost,GaussianNB,
                                LinearDiscriminantAnalysis,
                                QuadraticDiscriminantAnalysis,
                                SVM,MultinomialNaiveBayes}

                Regression {DecisionTree,LinearRegression,SVM,
                            KNeighbors,RandomForest,AdaBoost,
                            GradientBoosting,Bagging,ExtraTree}

        parameter : Classification:
                    [DecisionTree : None] [GaussianNB : None] [LinearDiscriminantAnalysis : None]
                    [QuadraticDiscriminantAnalysis : None]
                    [KNN : n_neighbors]                  5
                    [Logistic : penalty{ 'l1','l2'}]     'l2'
                    [RandomForest : n_estimators]        100
                    [GBDT : n_estimators]                100
                    [AdaBoost : n_estimators]            50
                    [SVM : kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} C ]    'rbf' 1.0
                    [MultinomialNaiveBayes : alpha]      1.0

                    Regression:
                    [DecisionTree : None] [LinearRegression : None]  [ExtraTree : None]
                    [SVM : kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} C ]    'rbf' 1.0
                    [KNeighbors : n_neighbors]        5
                    [RandomForest : n_estimators]     10
                    [AdaBoost : n_estimators]         50
                    [GradientBoosting : n_estimators] 100
                    [Bagging : n_estimators]          10
         ---Return  
        None
        ------------------------------------------------------"""
        if self.__typeLearning == 'Classification':
            self.__classifierModel(model,parameter)
        elif self.__typeLearning == 'Regression':
            self.__regressionModel(model,parameter)
        
        self.__model.fit(self.__xTrain,self.__yTrain)

    def assessmentModel(self,cvNum=5):
        y_pred = self.__model.predict(self.__xTest)
        from sklearn.model_selection import cross_val_score

        if self.__typeLearning == 'Regression':
            from sklearn.metrics import mean_squared_error,median_absolute_error,mean_squared_log_error,mean_absolute_error,explained_variance_score,r2_score

            #mean_squared_error
            print('MSE: \t\t',mean_squared_error(self.__yTest,y_pred))
            print('RMSE: \t\t',np.sqrt(mean_squared_error(self.__yTest,y_pred)))
            #print('MSE: ',np.mean((self.__yTest-y_pred)**2))

            #median_absolute_error
            #print('median: ',np.median(np.abs(self.__yTest-y_pred)))
            print('median: \t\t',median_absolute_error(self.__yTest,y_pred))
            
            #mean_absolute_error
            #print('MAE: ',np.mean(np.abs(self.__yTest-y_pred)))
            print('MAE: \t\t',mean_absolute_error(self.__yTest,y_pred))
            
            #mean_squared_log_error
            print('MSLE: \t\t',mean_squared_log_error(self.__yTest,y_pred))
            #print('MSLE: ',np.mean((np.log(self.__yTest+1)-np.log(y_pred+1))**2))
            
            #explained_variance_score
            print('explained_variance: \t\t',explained_variance_score(self.__yTest,y_pred))
            #print('explained_variance: ',1-np.var(self.__yTest-y_pred)/np.var(self.__yTest))
            
            #r2_score
            print('R2: \t\t',r2_score(self.__yTest,y_pred))
            #print('R2: ',1-(np.sum((self.__yTest-y_pred)**2))/np.sum((self.__yTest -np.mean(self.__yTest))**2))


            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='neg_mean_squared_error') 
            print('cv MSE mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='r2') 
            print('cv r2 mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='explained_variance') 
            print('cv explained_variance mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='neg_mean_squared_log_error') 
            print('cv MSLE mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='neg_mean_absolute_error') 
            print('cv MAE mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='neg_median_absolute_error') 
            print('cv median mean: \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='neg_root_mean_squared_error') 
            print('cv RMSE mean: \t',scoresval.mean())

        if self.__typeLearning == 'Classification':
            from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,average_precision_score
            print('准确率: \t\t',accuracy_score(self.__yTest,y_pred))
            print('准确率-Balanced: \t',balanced_accuracy_score(self.__yTest,y_pred))
            
            print('F1-micro: \t\t',f1_score(self.__yTest,y_pred,average='micro'))
            print('F1-macro: \t\t',f1_score(self.__yTest,y_pred,average='macro'))
            print('F1-weighted: \t\t',f1_score(self.__yTest,y_pred,average='weighted'))
            
            print('精确率-micro: \t',precision_score(self.__yTest,y_pred,average='micro'))
            print('精确率-macro: \t',precision_score(self.__yTest,y_pred,average='macro'))
            print('精确率-weighted:',precision_score(self.__yTest,y_pred,average='weighted'))

            print('召回率-micro: \t',recall_score(self.__yTest,y_pred,average='micro'))
            print('召回率-macro: \t',recall_score(self.__yTest,y_pred,average='macro'))
            print('召回率-weighted:',recall_score(self.__yTest,y_pred,average='weighted'))

            print('Cohen\'s Kappa: \t',cohen_kappa_score(self.__yTest,y_pred))
            
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='accuracy') 
            print('cv accuracy mean:  \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='balanced_accuracy') 
            print('cv balanced_accuracy mean:',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='f1_micro') 
            print('cv f1_micro mean:  \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='f1_macro') 
            print('cv f1_macro mean:  \t',scoresval.mean())
            scoresval = cross_val_score(self.__model,self.__xData,self.__yData,cv=cvNum,scoring='f1_weighted') 
            print('cv f1_weighted mean:  \t',scoresval.mean())

            from sklearn.metrics import classification_report
            print('分类报告: ','\n',classification_report(self.__yTest,y_pred))


if __name__ == '__main__':

    model = machineLearning('Classification','TOMO-IMRTNTCPtoManteia.xlsx','骨髓抑制分级', 'mean',3)
    model.dataPretreatment('StandardScaler')
    model.dataPretreatment('MinMaxScaler')
    model.labelMinMaxScaler()
    values = model.featureSelection('VarianceThreshold') 
    # values = model.featureSelection('chi2')

    model.featureDimensionalityReduction('PCA',10)

    model.trainTestSplit(0.2,randomState=21)
    model.trainingModel('RandomForest',{"n_estimators":7})
    model.assessmentModel(5)