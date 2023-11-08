from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载训练有素的自然语言模型
model = tf.keras.models.load_model('modelo_de_lenguaje_natural.h5')

app = Flask(__name__)

# 定义接受自然语言查询作为输入的路径
@app.route('/convertir_a_cypher', methods=['POST'])
def convertir_a_cypher():
    # 获取请求体的自然语言查询
    consulta = request.json['consulta']

    # 预处理查询
    consulta_preprocesada = preprocesar_consulta(consulta)

    # 使用自然语言模型在Cypher中生成查询
    consulta_cypher = generar_consulta_cypher(consulta_preprocesada)

    # 在Cypher中返回查询作为响应
    return {'consulta_cypher': consulta_cypher}

# 用自然语言预处理查询的函数
def preprocesar_consulta(consulta, tokenizer, max_length):
    """
    预处理查询并将其转换为数字表示
    可以用张量流处理。

    Args:
        consulta (str): 自然语言查询。
        tokenizer (Tokenizer): 将用于转换查询的标记化器
            用数字表示。
        max_length (int): 查询数字序列的最大长度。

    Returns:
        numpy array: 表示查询的形状数组(1,max_length)
        preprocesada.

    """
    # Tokenizar 咨询
    consulta_tokens = tokenizer.texts_to_sequences([consulta])
    
    # 调整数字序列，使其具有固定长度
    consulta_padded = pad_sequences(consulta_tokens, maxlen=max_length, padding='post', truncating='post')
    
    return np.array(consulta_padded)

# 使用自然语言模型在Cypher中生成查询的函数
def generar_consulta_cypher(modelo, preprocesador, consulta):
    """
    从Tensorflow模型的输出生成一个Cypher查询。

    Args:
        modelo (tensorflow.keras.Model): 用于生成Cypher查询的模型。
        preprocesador (callable): 在发送查询之前用于预处理查询的函数 模式。
        consulta (str): 自然语言查询。

    Returns:
        str: 表示模型生成的Cypher查询的字符串。

    """
    # 预处理查询
    consulta_procesada = preprocesador(consulta)
    
    # 从模型输出生成Cypher查询
    output = modelo.predict(consulta_procesada)
    output = np.argmax(output, axis=-1)
    cypher = ''.join([preprocesador.get_word_index()[w] for w in output[0] if w != 0])
    
    return cypher
