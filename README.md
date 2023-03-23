# XAI models applied to investing

Machine learning techniques have recently become the norm
for detecting patterns in financial markets. However, relying solely on
machine learning algorithms to make decisions can have negative consequences, especially in such a critical area as investing. Investing involves
making decisions about buying or selling securities based on various factors. Investors are compensated for the risk of holding securities, which
may decline in value between the purchase and the sale. Can an investor/trader explain clearly the intuition of his position on securities ?

This project proposes a machine learning approach powered by eXplainable Artificial Intelligence techniques integrated into a trading pipeline.
I first define a feature selection framework tailored for
stock returns direction forecasting that increases the accuracy of the employed
ML model and then I use the SHAP (SHapley Additive exPlanation) values to
evaluate the contributions of each individual feature to the overall upward and
downward movement logit probability.
