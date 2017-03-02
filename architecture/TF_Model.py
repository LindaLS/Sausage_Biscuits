import tensorflow as tf
import math
import sys

class TF_Model:

    # Constructor
    def __init__(self, model_directory):
        self.model_dir = model_directory
        self.weights = []

        self.activation_functions = []
        self.num_units = []
        self.partially_connected = []
        self.filter_widths = []
        self.filter_strides = []
        self.num_filters = []
        self.recurrent_ids = []

        # Read config file
        self.parse_config()

        # Create model
        self.create_model()

    def parse_config(self):
        f = open(self.model_dir + '/config.txt', 'r')

        current_line = 1
        prev_layer_units = 0
        for line in f:
            # Remove newline
            line = line.rstrip('\n')
            line = line.rstrip('\r')

            # Remove comments
            pos = line.find('#')
            if(pos != -1):
                line = line[:pos]

            # Remove whitespace
            line = "".join(line.split())

            # If not empty line, parse info
            if(len(line) > 0):
                prev_layer_units = self.parse_line(current_line, line, prev_layer_units)
                current_line = current_line + 1

    def parse_line(self, current_line, line, prev_layer_units):

        # Layer parameter initialization
        num_neurons = 0
        activation_function = ""
        partially_connected = False
        filter_width = 0
        filter_stride = 0
        num_filters = 0
        recurrent_id = 0

        # Error string
        err_str = 'Error in configuration file on line ' + str(current_line)

        params = line.split(',')
        for param in params:
            temp = param.split('=', 1)
            param_name = temp[0]
            param_value = temp[1]

            if(param_name == 'num_neurons'):
                num_neurons = int(param_value)
            elif(param_name == 'activation'):
                activation_function = param_value
            elif(param_name == 'connectivity'):
                if(param_value[0] == '['):
                    partially_connected = True
                    temp = param_value.lstrip('[').rstrip(']')
                    temp = temp.split(';')
                    filter_width = int(temp[0])
                    filter_stride = int(temp[1])
                    num_filters = int(temp[2])
                else:
                    sys.exit(err_str + ": invalid connectivity syntax")
            elif(param_name == 'recurrent_id'):
                recurrent_id = int(param_value)

        # Assign parameters
        if (num_neurons == 0 and (filter_width==0 or filter_stride==0 or num_filters==0) ):
            sys.exit(err_str + ": invalid layer connection specification")
        elif (activation_function == '' and current_line > 1):
            sys.exit(err_str + ": activation function must be specified")
        elif (activation_function != '' and current_line == 1):
            sys.exit(err_str + ": activation function must not be specified for the input layer")

        self.num_units.append(num_neurons)
        self.activation_functions.append(activation_function)
        self.partially_connected.append(partially_connected)
        self.filter_widths.append(filter_width)
        self.filter_strides.append(filter_stride)
        self.num_filters.append(num_filters)
        self.recurrent_ids.append(recurrent_id)

    def create_model(self):
        self.weights = []
        num_layers = len(self.num_units)

        for i in range(num_layers-1):
            input_num = self.num_units[i]
            output_num = self.num_units[i+1]
            filter_width = self.filter_widths[i+1]
            filter_stride = self.filter_strides[i+1]
            num_filters = self.num_filters[i+1]
            partially_connected = self.partially_connected[i+1]

            output_per_filter = 0
            if(partially_connected):
                output_per_filter = (input_num - filter_width) / filter_stride + 1
                output_num = int(output_per_filter * num_filters)
                self.num_units[i+1] = output_num

                init_weight = math.sqrt(3/(filter_width + output_num))

                W = tf.Variable(tf.random_normal([filter_width, 1, num_filters], mean=0.0, stddev=init_weight))
                b = tf.Variable(tf.random_normal([num_filters], mean=0.0, stddev=init_weight))
            else:
                init_weight = math.sqrt(3/(input_num + output_num))

                W = tf.Variable(tf.random_normal([input_num, output_num], mean=0.0, stddev=init_weight))
                b = tf.Variable(tf.random_normal([output_num], mean=0.0, stddev=init_weight))

            self.weights.append((W,b))

    def predict(self, x, dropout=1, layer=0):
        output_layer = len(self.num_units)
        if(layer != 0):
            output_layer = layer

        temp = tf.shape(x)
        num_inputs = temp[0]
        current = x
        for i in range(output_layer-1):
            partially_connected = self.partially_connected[i+1]
            activation_function = self.activation_functions[i+1]
            weights = self.weights[i]
            W = weights[0]
            b = weights[1]

            if (i == len(self.activation_functions)-1):
                current = tf.nn.dropout(current, dropout)
            if(partially_connected):
                filter_stride = self.filter_strides[i+1]
                conv = tf.nn.conv1d(tf.expand_dims(current,-1), W, stride=filter_stride, padding="VALID")
                z = tf.reshape(conv + b, [num_inputs, -1])
            else:
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
