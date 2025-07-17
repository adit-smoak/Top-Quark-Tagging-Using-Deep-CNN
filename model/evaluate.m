load('testDataNew.mat');
load('trainedModel90k.mat');      

test_labels = categorical(test_labels);

adsXTest = arrayDatastore(test_dataset, 'IterationDimension', 4);
adsFTest = arrayDatastore(test_features, 'IterationDimension', 4);
adsYTest = arrayDatastore(test_labels, 'IterationDimension', 1);
dsTest = combine(adsXTest, adsFTest, adsYTest);

YPred = classify(net, dsTest);
accuracy = mean(YPred == test_labels);
disp("Test Accuracy: " + accuracy);

figure;
confusionchart(test_labels, YPred);
title('Confusion Matrix on Test Set');
