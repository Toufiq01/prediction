EUR/USD Exchange Rate Prediction Using Machine Learning

Abstract— This research work explored and analyzed the use of machine learning technique such as linear regression on EUR/USD exchange rates in the global forex market to forecast the future movement and compare between the daily and hourly data prediction. For the reason of comparison, linear regression has been applied on both hourly and daily almost equivalent data sets of EUR/USD exchange rates and showed differentiation of the results. It has been observed that the percentage of accuracy of daily data prediction is greater than hourly data prediction at the testing stage.

Keywords—foreign exchange rate, EUR/USD, forex market, machine learning, linear regression.

Introduction 
Machine learning techniques are massive famous and widely used in the financial market. Foreign exchange (FX, forex or currency market) is also a large part of the financial market estimated with daily trade volume almost $5 trillion. The factors of price movement in the forex market have interacted in a complex relation. That’s why forex market prediction is ponderous. Predicting financial market is crucial for trader, investigators, economists, and analysts. Day by day, prediction methods are being developed for getting more accuracy. This paper has been considered as applying machine learning techniques in the currency market. EUR/USD is the most actively traded currency pair in the FX market. In this research work, we predict EUR/USD data series in different time frame by using the regression technique. Regression technique is generally used in linear data set, but in the nonlinear data set, it is used rarely [1]. This technique is applied in both hour and day time frame for accomplishing prediction of EUR/USD exchange rate and showing the accuracy and error of the prediction.
Literature Review
        Swagat Ranjit, Shruti Shrestha, Sital Subedi and Subarna Shakya [4] worked on a research task about the comparison of algorithms in foreign exchange rate prediction in 2018. They used some machine learning techniques such as the artificial neural network (ANN), the recurrent neural network (RNN) to develop a forecast model between NRs against three major currencies which are euro, pound sterling, and US dollar. Dr. Gu Wang and Dr. Joerg Oesterrider analyzed currency risk management predicting the EUR/USD exchange rate on April 26, 2018 [6]. They developed a linear regression model and fixing the error by using the momentum signal. Dinesh K. Sharma, H.S. Hota, and Richa Handa made a project about predicting exchange rate using regression techniques in 2017[1]. They compare regression technique with ensemble regression techniques for non-linear data and observe that diverse of MAPE values. Sitti Wetenriajeng Sidehabi, Indrabayu, and Sofyan Tandungan researched on Statistical and Machine Learning Approach Forex Prediction Based on Empirical Data at 2016 International Conference on Computational Intelligence and Cybernetics [8]. They used machine learning as Support Vector Machine (SVM) and a hybrid form of Genetic Algorithm-Neural Network (GA-NN) and compare this two method result. Konstantinos Theofilatos, Spiros Likothanassis, and Andreas Karathanasopoulos investigated modeling and trading the EUR/USD exchange rate using machine learning techniques in 2012[2]. They applied and compared with different types of machine learning techniques. Tadashi Iokibe, Shoji Murata and Masaya Koyama [7] worked and analyzed on the prediction of foreign exchange rate by local fuzzy reconstruction method on 22-25 Oct. 1995.
Data Set and Methodology
Dukascopy Bank SA provides historical price data feed for the different type of forex instrument for different time series. Frequently hourly data set was collected from June 8th, 2018 to December 8th, 2018 and daily data has collected from December 31th, 2007 to January 12th, 2019. Data sets are collected in excel sheet with some features such as open, high, low, close, volume. Saturday and Sunday are the weekly holidays in the foreign exchange market. This two days price movement of pairs remains to intermit and show the same price of Friday. Almost four thousands of price data are collected in diverse time in each data set or excel sheet.
      For accomplishing the prediction, all the coding were written in python language. Python 3.7 (32-bit) version has been used through IDLE or integrated development environment for python. Data has been transformed and manipulated as our liking and defined the features.
      Though generally in forex and stock exchange data sets are found as the rarely missing value, in the place of missing data -99,999 value has been placed. As preprocessing, the features have been normalized in the range of 1 to -1 for speeding up the processing time and getting a precious accuracy. The general formula is given as:

               	z=(x-min(x))/(max(x)-min(x))                                   (1)

      Where x is an original value, z is the normalized value.               Linear regression classifier has been used through machine learning library such as Scikit-Learn and trains the machine learning classifier. Then train the classifier and taken the data to test the classifier. After training and testing, forecast out has been done as taking the whole data and forecast out of the data. Then the scale method has been applied to the forecast data based on all the known data to standardize the range of independent variables or features of data. For standardization first need to find the standard deviation as follow:

                           δ = √(1/N ∑_(i=1)^N▒〖(x_i-x ̅)〗^2 )                                (2)

      Where x ̅ is the mean of all x value and then divide the subtracted value by its standard deviation and equation is given as:    

                                Z = (x-x ̅)/δ                                                  (3)
      Model Selection technique was applied for dynamic data partitioning to make the forecast more precise. Finally, we got both predictions hourly and daily for future EUR/USD market.
Result Analysis
Linear regression has been applied on the EUR/USD exchange rate and we got clear forecast and comparison. Here, Table I and II display the real price, predicted price and percentage error of EUR/USD exchange rate respectively basis on hourly and daily.

	ONE HOUR BASE PRICE
Date and Time(GMT)	Real Price	Predicted Price	Percentage Error
08.12.2018 00:00:00.000	1.13758	1.136094	0.001306282
08.12.2018 01:00:00.000	1.13758	1.136422	0.00101795
08.12.2018 02:00:00.000	1.13758	1.136317	0.001110252
08.12.2018 03:00:00.000	1.13758	1.136448	0.000995095
08.12.2018 04:00:00.000	1.13758	1.136108	0.001293975
08.12.2018 05:00:00.000	1.13758	1.135766	0.001594613
08.12.2018 06:00:00.000	1.13758	1.135702	0.001650873










	DAILY BASIS PRICE
Date and Time(GMT)	Real Price	Predicted Price	Percentage Error
14.12.2018 22:00:00.000	1.13073	1.144793	0.012437098
15.12.2018 22:00:00.000	1.13073	1.146939	0.014334987
16.12.2018 22:00:00.000	1.13048	1.148891	0.016286002
17.12.2018 22:00:00.000	1.13479	1.145834	0.009732197
18.12.2018 22:00:00.000	1.13619	1.140734	0.003999331
19.12.2018 22:00:00.000	1.13784	1.140398	0.002248119
20.12.2018 22:00:00.000	1.14468	1.140026	0.004065765


      Fig. 1, and Fig. 2, show the foresight in the different time frame. In Fig. 1, data and prediction are based on hourly time series and Fig.  2 shows daily basis graph. Analysis displayed that daily time series graph is sleeker than the hourly time series graph and also in the sector of accuracy, the daily time frame is ahead. But error finding analysis is different such as daily time series contain more error.



Fig. 1.    EUR/USD exchange movement and forecast for one hour time frame

 

Fig. 2:     EUR/USD exchange movement and forecast for one day time frame
     
 Table 3 explores the comparison between one-hour time frame result and daily time frame result respectively along with the calculation of MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error) and accuracy. This experiment has been accomplished by self-written Python code.


	COMPARATIVE RESULT ANALYSIS
	One Hour time frame	Daily time frame
Mean Absolute Error (MAE)	0.004260	0.034124
Mean Squared Error (MSE)	2.978e-05	0.002027
Root Mean Squared Error (RMSE)	0.005457	0.045032
Accuracy  	0.844388	0.882185


CONCLUSION AND FUTURE ENHANCEMENT
In this paper, we stand a learning framework and normalize the myriad data set of EUR/USD exchange rate. Then we applied machine learning technique named as linear regression on different time series of EUR/USD exchange rate of global forex market to compare the results in accuracy and various error method and got a different result for different time chart. The successful comparison of this paper also explore that for trading with more accuracy, daily data chart is better. In future, our intention is to use machine learning technique in financial other market such as stock market for making a better and safe trading.

References
[1] Dinesh K. Sharma, H.S. Hota, Richa Handa, “Prediction of foreign exchange rate using regression techniques”, Review of Business and Technology Research, Vol. 14, No. 1,2017,ISSN1941-9414.
[2] Konstantinos Theofilatos, Spiros Likothanassis, Andreas Karathanasopoulos, “Modeling and Trading the EUR/USD Exchange Rate Using Machine Learning Techniques”, ETASR - Engineering, Technology & Applied Science Research, Vol. 2, No. 5, 2012, 269-272.
[3] Kei Shioda, Shangkun Deng and Akito Sakurai, “Prediction of Foreign Exchange Market States with Support Vector Machine”, 2011 10th International Conference on Machine Learning and Applications.
 [4] Swagat Ranjit, Shruti Shrestha, Sital Subedi and Subarna Shakya,” Comparison of algorithms in Foreign Exchange Rate Prediction”, 2018 IEEE 3rd International Conference on Computing, Communication and Security (ICCCS), Kathmandu (Nepal).
[5] Christian L. Dunis and Mark Williams,“Modelling and Trading the EUR/USD Exchange Rate: Do Neural Network Models Perform Better?”,February 2002
[6] Dr. Gu Wang, Dr. Joerg Oesterrider, “Currency Risk Management Predicting the EUR/USD Exchange Rate”, April 26, 2018.
[7] Tadashi Iokibe, Shoji Murata and Masaya Koyama, “Prediction of Foreign Exchange Rate by Local Fuzzy Reconstruction Method”, 22-25 Oct. 1995
[8] Sitti Wetenriajeng Sidehabi, Indrabayu and Sofyan Tandungan, “Statistical and Machine Learning Approach in Forex Prediction Based on Empirical Data”, 2016





























 

