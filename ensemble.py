prediction1 = pd.read_pickle("data/arboretum-network.pkl")
prediction1.rename(index = str, columns = {"prediction" : "prediction_1"}, inplace = True)

prediction2 = pd.read_pickle("data/prediction_lgbm_2_1.pkl")
prediction2.rename(index = str, columns = {"prediction" : "prediction_2"}, inplace = True)

#prediction3 = pd.read_pickle("data/prediction_lgbm_5_1.pkl")
#prediction3.rename(index = str, columns = {"prediction" : "prediction_3"}, inplace = True)

prediction_mean = prediction1.merge(prediction2, on = ['order_id', 'product_id'], how = 'left')
#prediction_mean = prediction_mean.merge(prediction3, on = ['order_id', 'product_id'], how = 'left')

prediction_median = prediction_mean.copy()
prediction_log = prediction_mean.copy()


prediction_mean.loc[:, 'prediction'] = prediction_mean.loc[:, ['prediction_1', 'prediction_2', 'prediction_3']].mean(axis = 1)
prediction_median.loc[:, 'prediction'] = prediction_median.loc[:, ['prediction_1', 'prediction_2', 'prediction_3']].median(axis = 1)

prediction_log['prediction_1'] = np.log(prediction_log['prediction_1'].values)
prediction_log['prediction_2'] = np.log(prediction_log['prediction_2'].values)
prediction_log['prediction_3'] = np.log(prediction_log['prediction_3'].values)
prediction_log['prediction'] = prediction_log.loc[:, ['prediction_1', 'prediction_2', 'prediction_3']].mean(axis = 1)
prediction_log.loc[:, 'prediction'] = np.exp(prediction_log['prediction'].values)

prediction_mean[['product_id', 'order_id', 'prediction']].to_pickle('data/ensemble_mean_1.pkl')
prediction_median[['product_id', 'order_id', 'prediction']].to_pickle('data/ensemble_median_1.pkl')
prediction_log[['product_id', 'order_id', 'prediction']].to_pickle('data/ensemble_log_mean_1.pkl')


