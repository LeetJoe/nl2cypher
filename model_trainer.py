import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 加载训练和测试数据
def obtener_datos_entrenamiento():
    datos = pd.read_csv("data/datos_entrenamiento.csv")
    train = datos.sample(frac=0.8,random_state=200)
    test = datos.drop(train.index)
    return train, test

datos_entrenamiento, datos_prueba = obtener_datos_entrenamiento()

le = LabelEncoder()
datos_entrenamiento["cypher"] = le.fit_transform(datos_entrenamiento["cypher"])
datos_entrenamiento["cypher"] = tf.keras.utils.to_categorical(datos_entrenamiento["cypher"])

datos_prueba["cypher"] = le.fit_transform(datos_prueba["cypher"])
datos_prueba["cypher"] = tf.keras.utils.to_categorical(datos_prueba["cypher"])

# 预处理数据
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(datos_entrenamiento['text'])
datos_entrenamiento_tokens = tokenizer.texts_to_sequences(datos_entrenamiento['text'])
datos_prueba_tokens = tokenizer.texts_to_sequences(datos_prueba['text'])

# 向令牌序列添加填充
MAXLEN = 50
datos_entrenamiento_tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(datos_entrenamiento_tokens, maxlen=MAXLEN, padding='post')
datos_prueba_tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(datos_prueba_tokens, maxlen=MAXLEN, padding='post')

# 定义模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length=MAXLEN),
  tf.keras.layers.LSTM(128),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 训练模型
model.fit(datos_entrenamiento_tokens_padded, datos_entrenamiento['cypher'], epochs=10, validation_data=(datos_prueba_tokens_padded, datos_prueba['cypher']))

# 保存模型
model.save('modelo_de_lenguaje_natural.h5')

