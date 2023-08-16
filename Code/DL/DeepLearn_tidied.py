import nltk
import math
import time
import sklearn
import transformers
import contractions
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from tensorflow.python.ops.math_ops import truncate_div_eager_fallback
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from official.nlp import optimization

##### FUNCTIONS #####

def reverse_contraction(news):
    new_news = []
    for index, rows in news.iterrows():
        rows.dropna(inplace=True) 
        #removes contractions eg. "I'd like to" -> "I would like to"
        rows = rows.apply(contractions.fix)  
        new_news.append(rows)
    news = pd.DataFrame(new_news) 
    return news



def combine_and_reshape(news, dataset):
    stockdata = dataset.loc[:, [label_settings, second_input_settings]]
    combined_news = []
    for index, rows in news.iterrows():
        news.iloc[index] = rows
        label = stockdata.iat[index, 0]
        volume = stockdata.iat[index, 1]
        # Transform the 25xN table into a 1x25N table 
        for element in rows: # each element is one news title
            combined_news.append([element, volume, label]) 
    combined_news_df = pd.DataFrame(combined_news, columns=[first_input_settings, second_input_settings, label_settings])
    return combined_news_df


def to_tf_dataset_label(pd_dataframe):
    tf_dataset = tf.data.Dataset.from_tensor_slices((pd_dataframe[first_input_settings].values, pd_dataframe[label_settings].values))
    return tf_dataset

def to_tf_dataset(pd_dataframe):
    tf_dataset = tf.data.Dataset.from_tensor_slices(pd_dataframe)
    return tf_dataset

# Shuffle dataset
def shuffleDataframe(pd_dataframe):
    return pd_dataframe.sample(frac=1).reset_index(drop=True)

def split_data_fromDataframe_multiclass(pd_dataframe):
    pd_dataframe = shuffleDataframe(pd_dataframe)
    global num_classes, unique_secondinput
    num_classes= len(pd_dataframe[label_settings].value_counts())
    print(f"Num classes : {num_classes}")
    unique_secondinput = len(pd_dataframe[second_input_settings].value_counts())
    # One-Hot encode our labels
    label_categorical = tf.keras.utils.to_categorical(pd_dataframe[label_settings].values, num_classes=num_classes) 
    # Split data
    text_train, text_temp, label_train, label_temp = train_test_split(pd_dataframe[first_input_settings], label_categorical, test_size=0.2)
    text_test, text_val = np.array_split(text_temp, 2)
    label_test, label_val = np.array_split(label_temp, 2)
    volume_train, volume_temp = train_test_split(pd_dataframe[second_input_settings], test_size=0.2)
    volume_test, volume_val = np.array_split(volume_temp, 2)
    return text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val



def build_model_OG():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

def build_model_multi_labels():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

def build_model_multi_labels_cnn():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Reshape((-1, 1))(net)
    net = tf.keras.layers.Conv1D(32, 3, activation='relu')(net)
    net = tf.keras.layers.GlobalMaxPooling1D()(net)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

def build_model_multi_labels_with_hidden():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(20, activation=None, name='hidden_layer_1')(net)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


def build_model_multi_labels_two_inputs():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    text = outputs['pooled_output']
    number = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='number_input')
    net = tf.keras.layers.concatenate([text, number], axis=1)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model([text_input, number], net)

def build_model_multi_labels_two_inputs_fix():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    text = outputs['pooled_output']
    number = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='number_input')
    net = tf.keras.layers.concatenate([text, number], axis=1)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name='softmax_classifier')(net)
    return tf.keras.Model([text_input, number], net)

def build_model_multi_labels_two_inputs_v2():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    text = outputs['pooled_output']
    text = tf.keras.layers.Dropout(dropout_settings)(text)
    text = tf.keras.layers.Dense(3, activation=None, name='text_classifier')(text)
    
    number = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='number_input')
    number = tf.keras.layers.Dense(1, activation="softmax", name='text_classifier')(text)
    
    net = tf.keras.layers.multiply([text, number])
    net = tf.keras.layers.Dense(3, activation=None, name='classifier')(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name='softmax_classifier')(net)
    return tf.keras.Model([text_input, number], net)


def build_model_multi_labels_two_inputs_with_cnn():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    text = outputs['pooled_output']
    number = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='number_input')
    net = tf.keras.layers.concatenate([text, number], axis=1)
    net = tf.keras.layers.Reshape((-1, 1))(net)
    net = tf.keras.layers.Conv1D(32, 3, activation='relu')(net)
    net = tf.keras.layers.GlobalMaxPooling1D()(net)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model([text_input, number], net)

def build_model_multi_labels_two_inputs_with_hidden_layers():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    text = outputs['pooled_output']
    number = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='number_input')
    net = tf.keras.layers.concatenate([text, number], axis=1)
    net = tf.keras.layers.Dense(128, activation='relu', name='hidden_layer')(net)
    net = tf.keras.layers.Dropout(dropout_settings)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model([text_input, number], net)


def loss_and_metrics():
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True), tf.metrics.CategoricalAccuracy()
    # if non_binary_labeled == False:
    #     return tf.keras.losses.BinaryCrossentropy(from_logits=True), tf.metrics.BinaryAccuracy()
    # else:
    #     return tf.keras.losses.CategoricalCrossentropy(from_logits=True), tf.metrics.CategoricalAccuracy()
    
def callback_monitor():
    return "categorical_accuracy", "val_categorical_accuracy"
    # if non_binary_labeled == False:
    #     return "binary_accuracy", "val_binary_accuracy"
    # else:
    #     return "categorical_accuracy", "val_categorical_accuracy"
def inputs(text_train, text_test, text_val, volume_train, volume_test, volume_val):
    if multi_inputs == False:
        train_input = text_train
        test_input = text_test
        val_input = text_val
        return train_input, test_input, val_input 
    else:
        train_input = [text_train, volume_train]
        test_input = [text_test, volume_test]
        val_input = [text_val, volume_val]
        return train_input, test_input, val_input 
    
def train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val):
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalAccuracy()
    
    train_input, test_input, val_input = inputs(text_train, text_test, text_val, volume_train, volume_test, volume_val)
    # epochs
    epochs = epochs_settings
    steps_per_epoch = math.ceil(len(text_train) / batch_size_settings) #tf.data.experimental.cardinality(text_train).numpy() #would be the number of batches in total
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    
    # learning rate
    init_lr = 1e-5
    
    # set optimizer
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    
    #compile model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    
    
    # earlystop callbacl
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = callback_monitor()[0], 
                                                          patience = 15,
                                                          mode="auto",
                                                          restore_best_weights = True,
                                                          verbose=1,
                                                          min_delta=0.025)
    
    # train model
    history = model.fit(x=train_input,
                        y=label_train,
                        batch_size = batch_size_settings,
                        shuffle = True,
                        validation_data=(val_input, label_val),
                        epochs=epochs,
                        callbacks=[earlystop_callback])
            
            
    # save model
    model_path = model_folder_settings + "/" + model_version + "/"
    
    print(model_path)
    model.save(model_path + "model.h5", include_optimizer=False)
    

    tf.keras.utils.plot_model(model, show_layer_names=True, show_layer_activations=truncate_div_eager_fallback, to_file=(model_path + "model.png"), dpi=600)
    # model evaluation
    loss, accuracy = model.evaluate(test_input, label_test)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    
    #save test data to file
    pd.DataFrame(label_test).to_csv(model_path + "label_test.csv")
    pd.DataFrame(test_input).to_csv(model_path + "test_input.csv")
    pd.DataFrame(text_test).to_csv(model_path + "text_test.csv")
    
    predictions = model.predict(test_input)
    # print(predictions)
    # print(label_test)
    # print("Confusion Matrix")
    # label_names = np.unique(label_test, axis=0)
    # print(label_names)
    
    # cm = confusion_matrix(label_test, predictions)
    # sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    # plt.savefig(model_path + "confusion_matrix.png")
    # confusion_matrix = tf.math.confusion_matrix(label_test, predictions)
    # cm_np = confusion_matrix.numpy()
    # label_names = np.unique(label_test, axis=0)
    # print(label_names)
    # df_cm = pd.DataFrame(confusion_matrix.numpy(), index = [i for i in label_names],
    #                      columns = [i for i in label_names])
    # print(cm_np)
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    # plt.savefig(model_path + "confusion_matrix.png")
    
    pretrained_label_test = pretrained(text_test)
    loss, accuracy = model.evaluate(test_input, pretrained_label_test)
    print("Compare to pretrain model results")
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
   
    
    print(model.predict(test_input))
    
    history_dict = history.history

    acc = history_dict[callback_monitor()[0]]
    val_acc = history_dict[callback_monitor()[1]]
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.savefig(model_path + "training_history.png")
    #plt.show()
    return model
    
# pretrained stuff for evaluation
def pretrained(dataset):
    pretrained_model_name = "Jean-Baptiste/roberta-large-financial-news-sentiment-en"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = pipe(dataset.tolist())
    transformed_results = []
    for i in range(len(results)):
        if results[i]['label'] == "neutral":
            transformed_results.append(0)
        elif results[i]['label'] == "positive":
            transformed_results.append(1)
        else:
            transformed_results.append(-1)
    return tf.keras.utils.to_categorical(transformed_results, num_classes=num_classes)

def finbert(dataset):
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
    
    dataset_array = tf_dataset_to_array(dataset)
    
    new_dataset = []
    
    for i in range(0, len(dataset_array), 2):
        new_dataset.append(dataset_array[i])
        
    results = nlp(new_dataset)
    
    
    return results

def tf_dataset_to_array(dataset):
    dataset = dataset.unbatch()
    dataset_array = np.concatenate([batch for batch in dataset.as_numpy_iterator()])
    dataset_array = [x.decode('UTF-8') for x in dataset_array]
    return dataset_array
# pretrained stuff for evaluation  
    






####settings####
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'   
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
file_name = "Processed_Dataset_short"
default_model_folder_settings = "./Models/"
model_folder_settings = ""
model_version = ""
batch_size_settings = 32
epochs_settings = 1
dropout_settings = 0.5
non_binary_labeled = True # set to True, simply use categoricalcrossentrophy is acceptable
multi_inputs = True
first_input_settings = 'News'
second_input_settings = 'Vol_change' # or 'Volume'
label_settings = 'Label' #or 'Label_I' or 'Label_II'
####settings#### 

##initialize##   
num_classes = 0
unique_secondinput = 0
##### MAIN #####
dataset = pd.read_csv(file_name+".csv")
dataset.drop(columns=dataset.columns[0], axis=1, inplace=True)
news = dataset.iloc[:,1:26]
tokenized_news = []
# contraction fix
news = reverse_contraction(news)



######### Every Model Combination Run #########

########settings#######
non_binary_labeled = True
multi_inputs = True
second_input_settings = 'Vol_change' # or 'Volume'
label_settings = 'Label' #or 'Label_I' or 'Label_II'
num_of_inputs = "one_input"
non_binary_labeled = False
########settings#######

######### single input #########
multi_inputs = False
input_type = "Single_Inputs"
## binary
label_settings = 'Label'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_with_hidden()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_I
label_settings = 'Label_I'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_with_hidden()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_II
label_settings = 'Label_II'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_with_hidden()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {label_settings} {model_version}...")
model = build_model_multi_labels_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

######### multiple inputs #########
input_type = "Multiple_Inputs"
### Vol_change
multi_inputs = True
second_input_settings = "Vol_change"
## binary
label_settings = 'Label'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_I
label_settings = 'Label_I'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_II
label_settings = 'Label_II'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))


### Volume
multi_inputs = True
second_input_settings = "Volume"
## binary
label_settings = 'Label'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_I
label_settings = 'Label_I'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))

## non-binary Label_II
label_settings = 'Label_II'
comb_news = combine_and_reshape(news, dataset)
text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val = split_data_fromDataframe_multiclass(comb_news)
model_folder_settings = default_model_folder_settings + "/" + input_type + "/" +second_input_settings + "/" + label_settings + "/"
#normal
model_version = "normal_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#hidden
model_version = "hidden_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_hidden_layers()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
#cnn
model_version = "cnn_ver"
print(f"Build Model {input_type} {second_input_settings} {label_settings} {model_version}...")
model = build_model_multi_labels_two_inputs_with_cnn()
print(model.summary())
print(f"Train Model {num_of_inputs} {model_version}...")
start_time = time.time()
train_model(model, text_train, label_train, text_test, label_test, text_val, label_val, volume_train, volume_test, volume_val)
print("--- %s seconds ---" % (time.time() - start_time))
print("-----END-----")

