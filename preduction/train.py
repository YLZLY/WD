#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019-08-07 13:50 
# @Author : YJM
# @Site :  
# @File : train.py 
# @Software: PyCharm

import tensorflow as tf
import ssl
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")



def get_feature_column():
    """
    :return:wide feature column,deep feature column
    """
    #将连续特征获取出来
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hour_per_work = tf.feature_column.numeric_column("hour_per_work")

    #将离散出来的特征首先进行hash，hash得到的部分放入wide层，进行embedding，将embedding得到的部分放入deep层
    #hash_bucket_size=512的原因是离散出来的特征是不会超过512的
    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status",hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=512)
    relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=512)

    # 对连续值做离散化，这里使用的值都是按照自己的理解去分隔的，全都是自己定义的分割值
    # bucketized_column连续值离散化方法
    age_bucket = tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,45,50,55,60,65])
    gain_bucket = tf.feature_column.bucketized_column(capital_gain,boundaries=[0,1000,2000,3000,10000])
    loss_bucket = tf.feature_column.bucketized_column(capital_loss,boundaries=[0,1000,2000,3000,5000])

    #交叉特征
    cross_columns = [
        #年龄被分成了9段，收入被分成了4段，所以hash_bucket_size的结果就是9x4
        tf.feature_column.crossed_column([age_bucket,gain_bucket],hash_bucket_size=36),
        #收入和支出进行交叉，收入4段支出4段所以是16段
        tf.feature_column.crossed_column([gain_bucket,loss_bucket],hash_bucket_size=36),

    ]
    # 定义一个数据结构存储hash和离散化的特征
    base_columns = [work_class,education,marital_status,occupation,relationship,age_bucket,gain_bucket,loss_bucket]
    # W层的特征就是hash 加上 离散化 加上 交叉特征
    wide_columns = base_columns +cross_columns
    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hour_per_work,
        tf.feature_column.embedding_column(work_class,9),#选9的意义是2的9次方是512可以涵盖我们的哈希
        tf.feature_column.embedding_column(education, 9),
        tf.feature_column.embedding_column(marital_status, 9),
        tf.feature_column.embedding_column(occupation, 9),
        tf.feature_column.embedding_column(relationship, 9),

    ]
    return wide_columns,deep_columns


def build_model_estimator(wide_column,deep_column,model_folder):
    """

    :param wide_column:
    :param deep_column:
    :param model_export_floder:
    :return:model_es 模型实例，serving_input_fn辅助模型导出提供服务的函数
    """
    # DNNLinearCombinedClassifier就是wd模型的函数
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,#导入的是传入的参数
        linear_feature_columns=wide_column,#wide侧的特征
        # 优化器选用Ftrl，指定学习率是0.1，指定l2正则，正则化参数1.0
        linear_optimizer=tf.compat.v1.train.FtrlOptimizer(0.1,l2_regularization_strength=1.0),
        dnn_feature_columns = deep_column,
        # 指定优化器，指定学习率，指定正则化参数，不同优化器是为了参数迭代平稳，不要一次迭代太大，
        # 有的是控制学习率的方式，有的是控制参数梯度在本次迭代中影响的方式
        dnn_optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        # 隐层的维度
        # 隐层节点个数决定参数的整体维度
        # tf.feature_column.embedding_column(work_class, 9),  # 选9的意义是2的9次方是512可以涵盖我们的哈希
        # tf.feature_column.embedding_column(education, 9),
        # tf.feature_column.embedding_column(marital_status, 9),
        # tf.feature_column.embedding_column(occupation, 9),
        # tf.feature_column.embedding_column(relationship, 9),
        # 上面的这五个维度的特征有5x9个特征,45个特征与128个特征全连接，128与64全连接，依次类推
        # 用45*128+128*64+64*32+32*16，，共约16000个参数，总需要16000乘以100，也就是160万的训练数据
        # 但实际只有3万的数据，所以要重复采样，重复采样55倍
        dnn_hidden_units=[128,64,32,16]
    )
    # 固定打法原样写出来
    feature_column = wide_column + deep_column
    feature_space = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn=(tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_space))

    return  model_es,serving_input_fn

def input_fn(data_file,re_time,shuffle,batch_num,predict):
    """
    :param data_file: input data,train or test 训练文件或者测试文件
    :param re_time: time to repeat the data file 重复采样的次数
    :param shuffle: 布尔型 是否打乱数据
    :param batch_num: 随机梯度下降时，多少个样本我们更新参数一下
    :param predict: 布尔型，训练还是测试
    :return:
        两种情况：
            一：返回训练的feature和label
            二：返回测试的feature

    """
    _CSV_COLUMN_DEFAULTS = [[0],[''],[0],[''],[0],[''],[''],[''],[''],[''],
                            [0],[0],[0],[''],['']]
    _CSV_COLUMNS = [
        'age','workclass','fnlwgt','education','education_num',
        'marital_status','occupation','relationship','race','gender',
        'capital_gain','capital_loss','hour_per_work','native_country',
        'label'
    ]
    def parse_csv(value):
        columns = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS,columns))
        labels =features.pop('label')
        classes = tf.equal(labels,">50k")
        return features,classes

    def parse_csv_predict(value):
        columns = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        # 特征最终返回的是字典要注意key是特征名称 和稀疏featur可以理解成列表
        features = dict(zip(_CSV_COLUMNS,columns))
        labels =features.pop('label')
        return features
    # 读文件过滤第一行.skip(1).同样过滤掉有？的行
    # fp=open(data_file)
    # for line in fp:
    #     print(line)
    data_set = tf.data.TextLineDataset(data_file).skip(1)

    # data_set = tf.data.TextLineDataset(data_file)

    # Step-3: 创建数据迭代器
    # iterator = data_set.make_one_shot_iterator()
    #
    # # Step-4:
    # x = iterator.get_next()
    # with tf.Session() as sess:
    #     for i in range(4):
    #         print(sess.run(x))

    # 如果需要打乱就打乱
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict,num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv,num_parallel_calls=5)
    #     重采样
    data_set =  data_set.repeat(re_time)
    #     分割成batch
    data_set = data_set.batch(batch_num)
    return data_set

def train_wd_model(model_es,train_file,test_file,model_export_floder,serving_input_fn):
    """
    :param model_es: wd estimator
    :param train_file:
    :param test_file:
    :param model_export_floder:为提供服务所将模型导出到的文件夹
    :param serving_input_fn:辅助模型导出的函数
    :return:
    """
    total_run = 6
    for index in range(total_run):#重采样
        model_es.train(input_fn=lambda:input_fn(train_file,20,True,100,False))
        model_es.evaluate(input_fn=lambda:input_fn(test_file,1,False,100,False))
    model_es.export_saved_model(model_export_floder,serving_input_fn)

def get_test_label(test_file):
    """
    get label of test_file
    """
    if not os.path.exists(test_file):
        return []
    fp = open(test_file)
    linenum = 0
    test_label_list=[]
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        if "?" in line.strip():
            continue
        item = line.strip().split(",")
        label_str = item[-1]
        if label_str == ">50k":
            test_label_list.append(1)
        elif label_str == "<=50k":
            test_label_list.append(0)
        else:
            print("error")
    fp.close()
    return test_label_list

def test_model_performance(model_es,test_file):
    """
   test model auc in test data
    """
    test_label = get_test_label(test_file)
    result = model_es.predict(input_fn=lambda:input_fn(test_file,1,False,100,True))
    predict_list = []
    for one_res in result:
        if "probabilities" in one_res:
            predict_list.append(one_res["probabilities"][1])
    get_auc(predict_list,test_label)


def run_main(train_file,test_file,model_floder,model_export_floder):
    """
    :param train_file:
    :param test_file:
    :param model_floder: origin model floder to put train model
    :param model_export_floder: for tf serving
    """
    wide_column,deep_column = get_feature_column()
    model_es,serving_input_fn=build_model_estimator(wide_column,deep_column,model_floder)
    train_wd_model(model_es,train_file,test_file,model_export_floder,serving_input_fn)
    test_model_performance(model_es,test_file)

if __name__ == "__main__":
    run_main("../data/train.txt","../data/test.txt","../data/wd","../data/wd_export")



