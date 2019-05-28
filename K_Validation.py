'''
k折法训练模型并在新数据上测试结果
'''

k = 4
num_validation_samples = len(data) // k
np.shuffle(data)

validation_score = []
for fold in range(k):
    # 验证数据从第一块开始取
    validation_data = data[fold * num_validation_samples:
    (fold + 1) * num_validation_samples]
    # 训练数据取验证数据前面的块和后面的块
    training_data = data[:fold * num_validation_samples] +
                    data[(fold + 1) * num_validation_samples:]
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_score.append(validation_score)

average_score = np.average(validation_score)

# 在所有非测试数据上训练最终模型
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)