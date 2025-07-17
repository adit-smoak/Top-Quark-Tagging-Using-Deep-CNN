inputFileTrain = 'jets90000.parquet.gzip';
inputFileVal = 'valJets4000.parquet.gzip';
inputFileTest = 'testJets.parquet.gzip';

[train_dataset, train_features, train_labels] = mapping_approach(inputFileTrain);
save('trainDataNew90k.mat', 'train_dataset', 'train_features', 'train_labels', '-v7.3');

[val_dataset, val_features, val_labels] = mapping_approach(inputFileVal);
save('valDataNew.mat', 'val_dataset', 'val_features', 'val_labels', '-v7.3');

[test_dataset, test_features, test_labels] = mapping_approach(inputFileTest);
save('testDataNew.mat', 'test_dataset', 'test_features', 'test_labels', '-v7.3');