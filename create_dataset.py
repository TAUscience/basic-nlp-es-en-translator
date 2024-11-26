import tensorflow as tf
import tensorflow_text as tf_text

# Leer el archivo y separar las líneas
file_path = 'eng-spa.txt'

pairs = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Separar la línea en dos partes (en inglés y en español)
        parts = line.strip().split('\t')  # Asumiendo que las traducciones están separadas por tabuladores
        if len(parts) == 2:
            # Invertir el orden para español-inglés
            pairs.append((parts[1], parts[0]))  # (español, inglés)

# Convertir los pares en un tf.data.Dataset
def create_dataset(pairs):
    # Crear un Dataset a partir de los pares de texto
    dataset = tf.data.Dataset.from_generator(
        lambda: pairs, 
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # Texto en español
            tf.TensorSpec(shape=(), dtype=tf.string)   # Texto en inglés
        )
    )
    return dataset

# Crear el dataset completo
full_dataset = create_dataset(pairs)

# Mezclar el dataset
full_dataset = full_dataset.shuffle(buffer_size=len(pairs), seed=42)

# Dividir el dataset en train (80%), validation (5%) y test (15%)
train_size = int(0.8 * len(pairs))
val_size = int(0.05 * len(pairs))
test_size = len(pairs) - train_size - val_size

train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)

val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)
