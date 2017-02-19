import tensorflow as tf
import math

class TF_Model:

    # Constructor
    def __init__(self, model_directory):
        self.model_dir = model_directory
        self.weights = []

        self.activation_functions = []
        self.num_units = []

        # Read config file
        self.parse_config()

        # Create model
        self.create_model()

    def parse_config(self):
        f = open(self.model_dir + '/config.txt', 'r')
        parsing_arch = True
        for line in f:
            # Remove newline
            line = line.rstrip('\n')
            line = line.rstrip('\r')

            # Remove comments
            pos = line.find('#')
            if(pos != -1):
                line = line[:pos]

            # Remove whitespace
            line = line.strip()

            # If not empty line, parse info
            if(len(line) > 0):
                if(parsing_arch):
                    try:
                        self.num_units.append(int(line))
                    except:
                        parsing_arch = False
                        self.activation_functions.append(line)
                else:
                    self.activation_functions.append(line)

    def create_model(self):
        self.weights = []
        num_layers = len(self.num_units)

        for i in range(num_layers-1):
            input_num = self.num_units[i]
            output_num = self.num_units[i+1]
            init_weight = math.sqrt(6/(input_num + output_num))

            W = tf.Variable(tf.random_normal([input_num, output_num], mean=0.0, stddev=init_weight))
            b = tf.Variable(tf.random_normal([output_num], mean=0.0, stddev=init_weight))
            self.weights.append((W,b))

    def predict(self, x, dropout=1):
        current = x
        for i in range(len(self.activation_functions)):
            activation_function = self.activation_functions[i]
            weights = self.weights[i]
            W = weights[0]
            b = weights[1]

            if (i == len(self.activation_functions)-1):
                current = tf.nn.dropout(current, dropout)
            z = tf.matmul(current, W) + b

            if(activation_function == 'linear'):
                current = z
            elif(activation_function == 'relu'):
                current = tf.nn.relu(z)
            elif(activation_function == 'sigmoid'):
                current = tf.nn.sigmoid(z)
            elif(activation_function == 'softmax'):
                current = tf.nn.softmax(z)
            else:
                print(activation_function + ': activation function not supported!!')
                exit()

        return current

    def save(self, session, model_name):
        saver = tf.train.Saver()
        saver.save(session, self.model_dir + '/' + model_name + '.ckpt')

    def restore(self, session, model_name):
        saver = tf.train.Saver()
        saver.restore(session, self.model_dir + '/' + model_name + '.ckpt')
