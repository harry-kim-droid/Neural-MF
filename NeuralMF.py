#GMF with activation
from datetime import datetime
# 날짜와 시간을 나타내는 모듈
import tensorflow as tf
# 텐서플로우 호출
from tensorflow import keras
# 텐서플로우에서 케라스 호출
from functions.dataloader import dataloader
# functions.dataloader에서 dataloader호출
import matplotlib.pyplot as plt
# 그래프를 그리고 데이터를 시각화하는 라이브러리 호출
import pandas as pd
# 데이터 조작과 분석을 위한 라이브러리 호출
import argparse
# 명령줄 인터페이스를 파싱하는 라이브러리 호출
from GMF import GMF
# GMF 호출
from MLP import MLP
# MLP 호출
import numpy as np
# 배열이나 행렬을 다루는 라이브러리 호

#################### Arguments ####################
def parse_args():
    # 아래의 인자들을 파싱
    parser = argparse.ArgumentParser(description="NeuralMF.")
    # 파서는 ArgumentParser를 생성하고 NeuralMF에 대한 설명을 함
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    # 선택적으로 값을 받고 /dataset/을 기본값으로 설정, help는 사용자가 도움을 요청할때 보여줄 설명(input data path)
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    # 선택적으로 값을 받고 /ratings.csv/을 기본값으로 설정, help는 사용자가 도움을 요청할때 보여줄 설명(Choose a dataset)
    parser.add_argument('--layers', nargs='+', default=[64,32,16,8],
                        help='num of layers and nodes of each layer. embedding size is (2/1st layer) ')
    # 여러개의 값을 받고, [64,32,16,8]을 기본값으로 설정, help는 사용자가 도움을 요청할때 보여줄 설명(첫번째 레이어는 2배 즉128개를 가지고, 그 후의 레이어들은 32,16,8개의 노드를 가짐)
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    # 모델의 임베딩의 크기는 8을 기본값으로 설정하고 help는 사용자가 도움을 요청할때 보여줄 설명(MF모델의 임베딩 사이즈)
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    # 모델의 전체 데이터셋을 한 번 학습하는 것의 기본값을 10으로 설정하고, help는 사용자가 도움을 요청할때 보여줄 설명(에포크의 수)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    # 모델의 배치 크기는 32를 기본값으로 설정하고, help는 사용자가 도움을 요청할 때 보여줄 설명(배치 크기)
    parser.add_argument('--gmf_regs', type=float, default=0,
                        help='Regularization for MF embeddings.')
    # MF모델의 임베딩에 대한 정규화의 기본값을 0으로 설정하고, help는 사용자가 도움을 요청할 때 보여줄 설명(MF임베딩의 정규화)
    parser.add_argument('--mlp_regs', nargs='+', default=[0,0,0,0],
                        help="Regularization for user and item embeddings.")
    # MLP모델의 정규화의 기본값은 [0,0,0,0]으로 설정하고, help는 사용자가 도움을 요청할 때 보여줄 설명(사용자 및 아이템임베딩에 대한 정규화)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    # 모델의 학습률은 0.001을 기본값으로 설정하고, help는 사용자가 도움을 요청할 때 보여줄 설명(학습률)
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    # 옵티마이저는 adam으로 0개 또는 1개의 값을 받을 수 있음, help는 사용자가 도움을 요청할 때 보여줄 설명(옵티마이저로 adagrad, adam, rmsprop, sgd)
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    # 학습된 모델은 기본값이 1로 저장하고, help는 사용자가 도움을 요청할 때 보여줄 설명(훈련된 모델을 1or0으로 저장할지)
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    # 학습과정 에서 10번동안 개선되지 않을 때 조기종료, help는 사용자가 도움을 요청할 때 보여줄 설명(조기종료 페이션스)
    parser.add_argument('--pretrain_gmf', nargs='?', default='',
                        help='')
    # GMF의 사전 학습된 가중치를 선택적으로 0또는 1개의 값을 받을 수 있음
    parser.add_argument('--pretrain_mlp', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    # MLP모델의 사전학습된 가중치파일을 선택적으로 0또는 1개의 값을 받을 수 있음,help는 사용자가 도움을 요청할 때 보여줄 설명(사전학습된 MLP부분의 모델파일을 지정하며, 사전학습된 모델이 없으면 사용됨)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='pretrain trade off between GMF:alpha || MLP:1-alpha ')
    # MLP모델의 사전학습된 GMF와 MLP모델의 사전학습 가중치의 기본값을 0.5로 설정, help는 사용자가 도움을 요청할 때 보여줄 설명(사전학습된 GMF는 alpha,MLP는 1-alpha값)
    return parser.parse_args()
    # argparse라이브러리를 사용한 파싱인자들을 반환

class NeuralMF:
# NeuralMF 클래스 정

    def __init__(self,num_users,num_items,latent_features=8,layers = [64,32,16,8] ,gmf_regs=0,mlp_regs= [0,0,0,0]):
    # NeuralMF클래스는 사용자수, 아이템수, 잠재요인8개, 레이어수[64,32,16,8], GMF모델의 정규화값0, MLP레이어의 정규화값[0,0,0,0]
        self.num_users = num_users
        # NeuralMF클래스의 인스턴스가 생성될 때 num_users변수를 생성자로부터 전달된 값을 초기화
        self.num_items = num_items
        #gmf
        # NeuralMF클래스의 인스턴스가 생성될 때 num_items변수를 생성자로부터 전달된 값을 초기화
        self.latent_features = latent_features
        # NeuralMF클래스의 인스턴스가 생성될 때 latent_features변수를 생성자로부터 전달된 값을 초기화
        self.gmf_regs = gmf_regs
        #mlp
        # NeuralMF클래스의 인스턴스가 생성될 때 gmf_regs변수를 생성자로부터 전달된 값을 초기화
        self.layers = list(map(int,layers))
        # layers리스트 내의 각 요소를 정수로 변환하여 새로운 리스트로 할
        self.num_layers = len(layers)
        # layers리스트의 길이를 num_layers변수에 할당하여 저
        self.mlp_regs = list(map(float,mlp_regs))
        # MLP리스트의 각요소에 유리수로 변환하고이를 map객체에 다시 리스트로 반환하고 이를 self.mlp_regs에 할당하여 저장

        #inputs
        user_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        item_input = keras.layers.Input(shape=(1,),dtype = 'int32')

        #GMF_embedding_layer : embedding_size = num_factors
        user_embedding_gmf = keras.layers.Embedding(num_users, self.latent_features,
                                                embeddings_regularizer=keras.regularizers.l2(self.gmf_regs),
                                                    name = 'user_embedding_gmf')(user_input)
        item_embedding_gmf = keras.layers.Embedding(num_items, self.latent_features,
                                                embeddings_regularizer=keras.regularizers.l2(self.gmf_regs),
                                                    name = 'item_embedding_gmf')(item_input)
        user_latent_gmf = keras.layers.Flatten()(user_embedding_gmf)
        item_latent_gmf = keras.layers.Flatten()(item_embedding_gmf)

        result_gmf = keras.layers.Multiply()([user_latent_gmf,item_latent_gmf])

        #mlp_embedding layer : embedding_size = layer[0]/2
        user_embedding_mlp = keras.layers.Embedding(num_users,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.mlp_regs[0]),
                                                    name = 'user_embedding_mlp')(user_input)
        item_embedding_mlp = keras.layers.Embedding(num_items,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.mlp_regs[0]),
                                                    name = 'item_embedding_mlp')(item_input)

        user_latent_mlp = keras.layers.Flatten()(user_embedding_mlp)
        item_latent_mlp = keras.layers.Flatten()(item_embedding_mlp)

        result_mlp = keras.layers.concatenate([user_latent_mlp,item_latent_mlp])

        #mlp hidden layers : 1 ~ num_layer
        for index in range(self.num_layers):
            layer = keras.layers.Dense(layers[index],kernel_regularizer=keras.regularizers.l2(self.mlp_regs[index]),
                                       activation = keras.activations.relu,
                                       name = f'layer{index}')
            result_mlp =layer(result_mlp)


        #concat (gmf_result, mlp result)
        concat = keras.layers.concatenate([result_gmf,result_mlp])

        #predict rating
        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),
                                    name='output'
                                    )(concat)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])


    def get_model(self):
        model = self.model
        return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers,alpha):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding_gmf').set_weights(gmf_user_embeddings)
    model.get_layer('item_embedding_gmf').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding_mlp').set_weights(mlp_user_embeddings)
    model.get_layer('item_embedding_mlp').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(num_layers):
        mlp_layer_weights = mlp_model.get_layer(f'layer{i}').get_weights()

        model.get_layer(f'layer{i}').set_weights(mlp_layer_weights)


    # Prediction weights with hyper parameter 'alpha'
    gmf_output = gmf_model.get_layer('output').get_weights()
    mlp_output = mlp_model.get_layer('output').get_weights()


    pretrain_weights = np.concatenate((alpha * gmf_output[0], (1-alpha)*mlp_output[0]), axis=0)
    pretrain_bias = alpha * gmf_output[1] + (1-alpha)*mlp_output[1]
    model.get_layer('output').set_weights([pretrain_weights, pretrain_bias])
    return model


if __name__ =="__main__":
        #argparse
        args = parse_args()
        layers = args.layers
        num_factors = args.num_factors
        mlp_regs = args.mlp_regs
        gmf_regs = args.gmf_regs
        learner = args.learner
        learning_rate = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        patience = args.patience
        pretrain_gmf = args.pretrain_gmf
        pretrain_mlp = args.pretrain_mlp
        alpha = args.alpha

        #load datasets
        loader = dataloader(args.path + args.dataset)
        X_train,labels = loader.generate_trainset()
        X_test,test_labels =loader.generate_testset()

        #callbacks
        early_stop_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model_out_file = 'Pretrain/NeuralMF_%s.h5' % (datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
        model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

        #model
        model = NeuralMF(loader.num_users,loader.num_items,num_factors,layers,gmf_regs,mlp_regs).get_model()

        if pretrain_gmf != '' and pretrain_mlp != '':
            gmf_model = GMF(loader.num_users, loader.num_items,num_factors,gmf_regs).get_model()
            gmf_model.load_weights(pretrain_gmf)
            mlp_model = MLP(loader.num_users, loader.num_items, layers, mlp_regs).get_model()
            mlp_model.load_weights(pretrain_mlp)
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers), alpha =alpha)
            print(f"Load pretrained GMF ({pretrain_gmf}) and MLP ({pretrain_mlp}) models done. ")



        if learner.lower() == "adagrad":
            model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate), loss='mse')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), loss='mse')
        elif learner.lower() == "adam":
            model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        else:
            model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='mse')
        #train
        if args.out:
            history = model.fit([X_train[:,0],X_train[:,1]],labels,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=([X_test[:,0],X_test[:,1]],test_labels),
                                callbacks=[early_stop_cb,
                                           model_check_cb]
                      )
        else :
            history = model.fit([X_train[:, 0], X_train[:, 1]], labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=([X_test[:, 0], X_test[:, 1]], test_labels),
                                callbacks=[early_stop_cb]
                                )

        pd.DataFrame(history.history).plot(figsize= (8,5))
        plt.show()
        test_sample = X_test[:10]
        test_sample_label = test_labels[:10]
        print(model.predict([test_sample[:,0],test_sample[:,1]]))
        print(test_sample_label)

