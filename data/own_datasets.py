import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(name_ds):
    """
    Carga un dataset desde un archivo CSV (sin encabezados) y lo divide en train, validation y test.
    
    Args:
        file_path (str): Ruta al archivo CSV.
    
    Returns:
        dict: Diccionario con las particiones 'train', 'validation', 'test',
              cada una es un `tf.data.Dataset`.
    """
    # Cargar datos y asignar nombres de columnas
    file_path = f"data/{name_ds}.tsv"
    data = pd.read_csv(file_path, sep="\t", names=["input", "target"])
    assert "input" in data.columns and "target" in data.columns, \
        "El archivo debe tener columnas 'input' y 'target'"

    # Dividir en train, validation y test
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Crear datasets de TensorFlow
    def create_tf_dataset(dataframe):
        inputs = dataframe["input"].values
        targets = dataframe["target"].values
        return tf.data.Dataset.from_tensor_slices((inputs, targets))

    train_dataset = create_tf_dataset(train_data)
    val_dataset = create_tf_dataset(val_data)
    test_dataset = create_tf_dataset(test_data)

    # Prefetch para optimización
    dataset = {
        "train": train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE),
        "validation": val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE),
        "test": test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE),
    }
    return dataset
