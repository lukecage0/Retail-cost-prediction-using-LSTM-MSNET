# LSTM-MSNET
Implementation of LSTM-MSNet: Leveraging Forecasts on Sets of Related Time Series with Multiple Seasonal Patterns by  kasungayan
https://arxiv.org/pdf/1909.04293.pdf
Used Deseasonalizaton approach

# About the Dataset:
For Training Use Hourly-train.csv : https://www.kaggle.com/yogesh94/m4-forecasting-competition-dataset?select=Hourly-train.csv
For Testing Use Hourly-test.csv: https://www.kaggle.com/yogesh94/m4-forecasting-competition-dataset?select=Hourly-test.csv

﻿
One of our main challenges comprise of monitoring and analyzing client's business metrics in real time for instant detection of the incidents that may impact their revenue. 
One subpart of this challenge is Autonomous Forecasting which anticipate demand and business results so that clients operation is optimized for every future scenario. 
The Problem Statement presented below highlights a paper which attempts to solve Autonomous Forecasting 
Problem Statement
LSTM-MSNet Overview
﻿
In this paper, the author proposes Long Short-Term Memory Multi-Seasonal Net (LSTM-MSNet), a decomposition based, unified prediction framework to forecast time series with multiple seasonal patterns. 
The paper deals with Time Series Forecasting and proposes a method which takes inspiration from both the Statistical and Deep Learning World.  The Author of the paper evaluates their method on three Public Time Series Datasets and displayed quite promising results on all three.
﻿
Breaking Down The Paper
https://storage.googleapis.com/slite-api-files-production/files/0667aeba-8911-41ae-99d3-3cb43a01e99e/image.png
﻿
LSTM-MSNet can be divided into 4 primary parts:
﻿
1) Time Series Pre-processing:
Normalisation : For normalizing the Time Series Data, author proposes the use of mean-scale
 transformation strategy, which uses the mean of a time series
 as the scaling factor. This scaling strategy can be deﬁned as
 follows:
﻿
https://storage.googleapis.com/slite-api-files-production/files/b4f4f758-4ce2-4310-a526-82f5dc63df03/image.png
Here, xi ,normalized represents the normalized observation, and k represents the number of observations of time series i.
Variance Stabilization Layer : After normalizing the time series, they stabilize the variance in the group of time series by transforming each time series to a logarithmic scale. The transformation can be defined in the following way: 
https://storage.googleapis.com/slite-api-files-production/files/975e4c7c-7f68-4f88-b6ca-687e79725a15/image.png
Moving Window Transformation : As a preprocessing step, they transform the past observations of time series (Xi) into multiple pairs of input and output frames using a Moving Window (MW) strategy. In summary, the MW strategy converts a time series Xi of length K into (K − n − m) records, where each record has an amount of (m + n) observations. Here, m refers to the length of the output window (Forecast Period), and n is the length of the input window (Look back Period). These frames are generated according to the Multi-Input Multi-Output (MIMO) principle used in multi-step forecasting, which directly predicts all the future observations up to the intended forecasting horizon.
https://storage.googleapis.com/slite-api-files-production/files/c1abbb30-7f07-49bc-b96e-0ea85b4a02de/image.png
The input window or the Lookback Period = n* output window or Forecast Period, with n being 1.5 in the paper. A very good example for understanding MIMO is:
https://storage.googleapis.com/slite-api-files-production/files/751ff70c-9209-46f4-8274-87774c4ccc56/image.png
﻿
2)  Seasonal Decomposition 
When modelling seasonal time series with NNs, many studies suggest applying a prior seasonal adjustment, i.e., de-seasonalization to the time series. The main intention of this approach is to minimize the complexity of the original time series by  detaching the multi-seasonal components from a time series, and thereby reducing the subsequent effort of the NN’s learning process. Here, Multi-seasonal components refer to the repeating patterns that exist in a time series and that may change slowly over time
The Author Proposes 5 Methods for Seasonal Decomposition : 
Multiple STL Decomposition (MSTL)
Seasonal-Trend decomposition by Regression (STR)
Trigonometric, Box-Cox, ARMA, Trend, Seasonal (TBATS)
Prophet
 Fourier Transformation
It is required by the applicant to implement any one of the above mentioned techniques.
https://storage.googleapis.com/slite-api-files-production/files/57a7a600-16fa-4444-977d-9398112b48b9/image.png
﻿
﻿
3) Training Paradigms
The Author proposed two methods for training the LSTM model:
Deseasonalised Approach (DS): This approach uses seasonally adjusted time series as moving window patches to train the LSTM-MSNet. Since the seasonal components are not included in DS for the training procedure, a reseasonalisation technique is later introduced in the Post-processing layer of LSTM-MSNet to ascertain the corresponding multiple seasonal components of the time series.
Seasonal Exogenous Approach (SE): This second approach uses the output of the pre-processing layer, together with the seasonal components extracted from the multi-seasonal decomposition as external variables. Here, in addition to the normalized time series (without the deseasonalisation phase), the seasonal components relevant to the last observation of the input window are used as exogenous variables in each input window. As the original components of the time series are used in the training phase of SE, the LSTM-MSNet is expected to forecast all the components of a time series, including the relevant multi-seasonal patterns. Therefore, a reseasonalisation stage is not required by SE.
﻿
In summary, DS supplements the LSTM-MSNet by excluding the seasonal factors in the LSTM-MSNet training procedure. This essentially minimises the overall training complexity of the LSTM-MSNet. In contrast, SE supplements LSTM-MSNet in the form of exogenous variables that assist modelling the seasonal trajectories of a time series.
﻿
Fortunately , it is required by the applicant to just implement Deseasonalised Approach (DS), feel free to also implement Seasonal Exogenous Approach (SE), though its not a mandatory criteria.
﻿
﻿
https://storage.googleapis.com/slite-api-files-production/files/c2cae6d9-0812-4c02-91bf-bab82a964fc8/image.png
﻿
LSTM Learning Scheme
As highlighted earlier, the author uses the past observations of time series Xi , in the form of input and output windows to train the LSTM-MSNet. The author uses the LSTM model mentioned in this paper: https://arxiv.org/pdf/1909.00590.pdf, feel free to use any LSTM implementation as long as it is working in the right way.
﻿
https://storage.googleapis.com/slite-api-files-production/files/de07b98f-10c3-4dfb-9bd0-a3644ed4d501/image.png
﻿
Loss Function 
The author uses the L1-norm, as the primary learning objective function, which essentially minimizes the absolute differences between the target values and the estimated values. They also include an L2-regularization term to minimize possible over fitting of the network
﻿
https://storage.googleapis.com/slite-api-files-production/files/4d90dec6-5e72-4d94-96ee-db4789164b81/image.png
4) Post-processing Layer:
The reseasonalisation and renormalisation is the main component of the post processing layer in LSTM-MSNet. Here, in the reseasonalisation stage, the relevant seasonal components of the time series are added to the forecasts generated by the LSTM. This is computed by repeating the last seasonal components of the time series to the intended forecast horizon. Next, in the renormalisation phase, the generated forecasts are back-transformed to their original scale by adding back the corresponding local normalization factor, and taking the exponent of the values. The final forecasts are obtained by multiplying this vector by the scaling factor used for the normalization process. 
﻿
