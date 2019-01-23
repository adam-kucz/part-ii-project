"""TODO: describe"""
import argparse
import tensorflow as tf

import identifier_type_data as data

LEARNING_RATE: float = 0.1


def my_model(features, labels, mode, params):
    """TODO: describe, 1D CNN with ... ."""

    # Create input layer.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # Create 1d convolutional layers.
    for conv_params in params['convolutional']:
        net = tf.layers.conv1d(inputs=net,
                               filters=conv_params['filters'],
                               kernel_size=conv_params['kernel_size'],
                               padding='valid',
                               use_bias=False,
                               activation=tf.nn.relu)

    # Create dense layers.
    for dense_params in params['dense']:
        net = tf.layers.dense(inputs=net,
                              units=dense_params['units'],
                              activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {
        'accuracy': accuracy
    }
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN  # nosec

    optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    """TODO: describe"""
    # Get program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,
                        help='number of training steps')
    args = parser.parse_args(argv[1:])

    # Prepare and fetch the data
    data_loader = data.DataLoader()
    data_loader.load_data()
    
    (train_x, train_y), (validate_x, validate_y) = data_loader.get_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 5 hidden layer CNN
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # 3 convolutional layers
            'convolutional': [{'filters': 32, 'kernel_size': 5},
                              {'filters': 32, 'kernel_size': 5},
                              {'filters': 32, 'kernel_size': 3}],
            # 2 dense layers
            'dense': [64, 64],
            # Number of classes determined by data
            'n_classes': data_loader.get_num_classes()
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda: data_loader.train_input_fn(train_x, train_y,
                                                    args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: data_loader.eval_input_fn(validate_x, validate_y,
                                                   args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
