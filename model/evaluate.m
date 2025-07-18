load('testDataNew.mat');
load('trainedModel.mat');  

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

YPredScores = predict(net, dsTest); 
scorePosClass = YPredScores(:, 1);
labelsLogical = (double(test_labels) == 1);

[fpRate, tpRate, T, AUC] = perfcurve(labelsLogical, scorePosClass, true);

figure;
plot(fpRate, tpRate, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC, '%.3f') ')']);
grid on;
