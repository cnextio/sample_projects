import plotly.express as px
import CycDataFrame
import sys
import pandas as pd

df = pd.DataFrame()

print('Hello')
for i in range(10):
    print("hello")

df = CycDataFrame('data/machine-simulation/21549286_out.csv')
df.drop(index=0, inplace=True)
df = pd.DataFrame()

df = CycDataFrame('data/housing_data/data.csv')
df['LotFrontage'] = df['LotFrontage'].fillna(method='ffill')

#! df 'LotFrontage' vs 'MSZoning', 'HouseStyle'

df = CycDataFrame(px.data.gapminder())
df['year_str']= pd.to_datetime(df['year'], format='%Y').dt.strftime('%Y')
#! df 'lifeExp' vs 'gdpPercap'
px.scatter(df, x='gdpPercap', y='lifeExp')

df = px.data.gapminder()
px.line(df, y='gdpPercap', x='lifeExp', color='country')

from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
lnl = LearningWithNoisyLabels(clf=LogisticRegression())
lnl.fit(X=X_train_data, s=train_noisy_labels)
# Estimate the predictions you would have gotten by training with *no* label errors.
predicted_test_labels = lnl.predict(X_test)

from sklearn import datasets
# iris = CycDataFrame(datasets.load_iris().data)
iris = datasets.load_iris().data
import numpy as np
iris_label = datasets.load_iris().target
iris_label = iris_label.reshape((150,1))
iris = np.hstack((iris, iris_label))

iris.shape
df = pd.DataFrame(data=iris, columns=["column1", "column2", "column3", "column4", 'label'])
df = CycDataFrame(df)

#! df 'column3' vs 'column1','label'
px.scatter(df, x='column1', y='column3', color='label')

import plotly.express as px
df = px.data.gapminder().query("continent == 'Oceania'")
px.line(df, x='year', y='lifeExp', color='country', markers=True)


import mlflow
import mlflow.tensorflow
import tensorflow as tf
from send2trash import send2trash
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

MLFLOW_TRACKING_URL = '.mlflow'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
mlflow.set_experiment('MNIST')

MODEL_TRAINING_LOG_DIR = "logs"
CHECKPOINT_DIR = os.path.join(MODEL_TRAINING_LOG_DIR, "checkpoints")
if os.path.exists(CHECKPOINT_DIR):
    send2trash(CHECKPOINT_DIR)
os.mkdir(CHECKPOINT_DIR)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "cnext_{epoch:d}.hdf5"),
    save_weights_only=False,
    verbose=1,
    save_best_only=False,
    save_freq='epoch'
)

def train():
    model.fit(x_train, y_train, epochs=3, callbacks=[cp_callback])    

mlflow.tensorflow.autolog()
with mlflow.start_run():
    model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])
    mlflow.log_artifacts(CHECKPOINT_DIR, 'checkpoints')
mlflow.end_run()

import netron
MODEL_PATH = os.path.join(MODEL_TRAINING_LOG_DIR, 'model.h5')
model.save(MODEL_PATH)
netron.stop(('localhost', 8080))
netron.start(MODEL_PATH, 8080)

import tensorflow as tf
import tensorflow_datasets as tfds

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

BUFFER_SIZE = 10 # Use a much larger value for real code
BATCH_SIZE = 64
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 5

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_data = mnist_test.map(scale).batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(0.001)

@tf.function
def train_step(model, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        regularization_loss=tf.math.add_n(model.losses)
        pred_loss=loss_fn(labels, predictions)
        total_loss=pred_loss + regularization_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss


def train(model, optimizer, dataset, log_freq=10):
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    for images, labels in dataset:        
        # print(optimizer.iterations)
        loss = train_step(model, optimizer, images, labels)                
        avg_loss.update_state(loss)      
        if tf.equal(optimizer.iterations % log_freq, 0):  
            # print(optimizer.iterations.numpy())
            tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
            avg_loss.reset_states()


def test(model, test_x, test_y, step_num):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    loss = loss_fn(model(test_x, training=False), test_y)
    tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

mlflow.set_tracking_uri('/Users/bachbui/works/cycai/cnext-working-dir/Skywalker/.mlflow')
mlflow.set_experiment('Skywalker-main.py')
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name='Run_2'):
# with train_summary_writer.as_default():
# train(model, optimizer, train_data, log_freq=1)
    for images, labels in train_data:      
        loss = train_step(model, optimizer, images, labels)  
        tf.summary.scalar('loss', loss, step=optimizer.iterations)
# mlflow.end_run()


# with test_summary_writer.as_default():
#     test(model, test_x, test_y, optimizer.iterations)

tracking_uri = '/Users/bachbui/works/cycai/cnext-working-dir/Skywalker/.mlflow' 
from mlflow.tracking.client import MlflowClient
import pandas as pd
mlFlowClient = mlflow.tracking.MlflowClient(tracking_uri)
# run = mlFlowClient.get_run('4b520b77779e4a6b91470649d08ce31e')
print(mlFlowClient.list_artifacts('4b520b77779e4a6b91470649d08ce31e', 'checkpoints/cnext_1.hdf5'))
local_path = mlFlowClient.download_artifacts('4b520b77779e4a6b91470649d08ce31e', "checkpoints/cnext_1.hdf5", '.cnext')
local_path

import re
res = re.search(r'(?<=cnext_)\d+', 'checkpoints/cnext_1.hdf5')
print(res.group(0))

run.data.metrics['loss']
run_ids = ['9e197bee5414440b93ae1b1a002f7fbe', '12403412de5c48ac8e0be547f70ee93f']
metrics_data = {}
metrics_df = pd.DataFrame()
metric_keys = None
for id in run_ids:
    run = mlFlowClient.get_run(id)
    metric_keys = run.data.metrics.keys() 
    for metric in metric_keys:
        metric_history = mlFlowClient.get_metric_history(id, metric)
        if metric not in metrics_data:
            metrics_data[metric] = {}
        # metrics_data[metric][id] = {'value': [m.value for m in metric_history], 
        #                             'step': [m.step for m in metric_history], 
        #                             'timestamp': [m.timestamp for m in metric_history]}
        # print(id, metrics_data)
            metrics_data[metric][id] = [m.value for m in metric_history]
            metrics_data['index'] = {'step': [m.step for m in metric_history]}

metrics_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in metrics_data[metric].items()]), 
                            index=metrics_data['index']['step'])
# for metric in metric_keys:
#     metrics_data[metric] = {(outerKey, innerKey): values for outerKey, innerDict in metrics_data[metric].items() for innerKey, values in innerDict.items()}
    # metrics_df = pd.DataFrame(dict([(index,run_metric) for index,run_metric in metrics_data[metric].items()]))
print(metrics_df.index)

metrics_data[metric].items()

import os
os.getcwd()


import plotly.graph_objects as go
import numpy as np
np.random.seed(1)

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N) - 5

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                    mode='markers',
                    name='markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                    mode='lines+markers',
                    name='lines+markers'))
fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                    mode='lines',
                    name='lines'))

fig