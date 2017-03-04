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
        self.layer_reference = []
        self.weight_inverted = []

        # Read config file
        self.parse_config()

        # Create model
        self.create_model()

    def parse_config(self):
        f = open(self.model_dir + '/config.txt', 'r')

        current_line = 1
        is_first_line = True
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
                self.parse_line(current_line, line, is_first_line)
                is_first_line = False
        
            current_line = current_line + 1

    def parse_line(self, current_line, line, is_first_line):

        # Layer parameter initialization
        num_neurons = 0
        activation_function = ""
        partially_connected = False
        filter_width = 0
        filter_stride = 0
        num_filters = 0
        repeated_layer = 0
        invert_layer = 0
        weight_inverted = False

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
            elif(param_name == 'repeat_layer'):
                repeated_layer = int(param_value)
            elif(param_name == 'invert_layer'):
                repeated_layer = int(param_value)
                weight_inverted = True

        # Assign parameters
        if (num_neurons == 0 and (filter_width==0 or filter_stride==0 or num_filters==0)  and repeated_layer==0):
            sys.exit(err_str + ": invalid layer connection specification")
        elif (activation_function == '' and not is_first_line):
            sys.exit(err_str + ": activation function must be specified")
        elif (activation_function != '' and is_first_line):
            sys.exit(err_str + ": activation function must not be specified for the input layer")

        self.num_units.append(num_neurons)
        self.activation_functions.append(activation_function)
        self.partially_connected.append(partially_connected)
        self.filter_widths.append(filter_width)
        self.filter_strides.append(filter_stride)
        self.num_filters.append(num_filters)
        self.layer_reference.append(repeated_layer)
        self.weight_inverted.append(weight_inverted)

    def create_model(self):
        self.weights = []
        weights_exist = []

        num_layers = len(self.num_units)
        
        # Update IO sizes
        self.layer_reference[0] = 1
        for connections in range(num_layers-1):
            current_connection = connections+1
            input_layer_index = current_connection - 1
            output_layer_index = current_connection

            weight_exist = True
            layer_reference = self.layer_reference[output_layer_index]
            weight_inverted = self.weight_inverted[current_connection]

            if(layer_reference == 0):
                self.layer_reference[output_layer_index] = output_layer_index+1
            else:
                self.num_units[output_layer_index] = self.num_units[layer_reference-1]
                weight_exist = False

                self.partially_connected[output_layer_index] = self.partially_connected[layer_reference-1]
                self.partially_connected[output_layer_index] = self.filter_widths[layer_reference-1]
                self.partially_connected[output_layer_index] = self.filter_strides[layer_reference-1]
                self.partially_connected[output_layer_index] = self.filter_strides[layer_reference-1]

            if(weight_inverted):
                num_output = self.num_units[layer_reference-1]
                num_input = self.num_units[layer_reference-2]
                self.num_units[output_layer_index] = num_input
                self.num_units[input_layer_index] = num_output


            partially_connected = self.partially_connected[current_connection]
            if(partially_connected):
                input_num = self.num_units[input_layer_index]
                output_num = self.num_units[output_layer_index]
                filter_width = self.filter_widths[current_connection]
                filter_stride = self.filter_strides[current_connection]
                num_filters = self.num_filters[current_connection]
                output_per_filter = (input_num - filter_width) / filter_stride + 1
                output_num = int(output_per_filter * num_filters)
                self.num_units[current_connection] = output_num

            weights_exist.append(weight_exist)

        # Create weights
        for connections in range(num_layers-1):
            current_connection = connections+1
            input_layer_index = current_connection - 1
            output_layer_index = current_connection

            weight_exist = weights_exist[input_layer_index]
            input_num = self.num_units[input_layer_index]
            output_num = self.num_units[output_layer_index]
            filter_width = self.filter_widths[current_connection]
            filter_stride = self.filter_strides[current_connection]
            num_filters = self.num_filters[current_connection]
            partially_connected = self.partially_connected[current_connection]
            weight_inverted = self.weight_inverted[current_connection]

            W = 0
            b = tf.Variable(tf.zeros([output_num]))
            if(weight_exist):
                output_per_filter = 0
                if(partially_connected):
                    output_per_filter = (input_num - filter_width) / filter_stride + 1
                    output_num = int(output_per_filter * num_filters)
                    self.num_units[current_connection] = output_num

                    init_weight = math.sqrt(3/(filter_width + output_num))

                    W = tf.Variable(tf.random_normal([filter_width, 1, num_filters], mean=0.0, stddev=init_weight))
                else:
                    init_weight = math.sqrt(3/(input_num + output_num))
                    W = tf.Variable(tf.random_normal([input_num, output_num], mean=0.0, stddev=init_weight))
                    

            self.weights.append((W,b))

    def predict(self, x, dropout=1, layer=0):
        output_layer = len(self.num_units)
        if(layer != 0):
            output_layer = layer

        temp = tf.shape(x)
        num_inputs = temp[0]
        current = x

        for i in range(output_layer-1):
            current_connection = i+1
            input_layer_index = i
            output_layer_index = i+1

            partially_connected = self.partially_connected[current_connection]
            activation_function = self.activation_functions[current_connection]
            weight_inverted = self.weight_inverted[current_connection]

            weights = self.weights[current_connection-1]
            b = weights[1]
            W = 0
            if(not weight_inverted):
                weights = self.weights[self.layer_reference[output_layer_index]-2]
                W = weights[0]
            else:
                weights = self.weights[self.layer_reference[output_layer_index]-2]
                W = tf.transpose(weights[0])

            if (i == len(self.activation_functions)-1):
                current = tf.nn.dropout(current, dropout)
            if(partially_connected):
                filter_stride = self.filter_strides[i+1]
                conv = tf.nn.conv1d(tf.expand_dims(current,-1), W, stride=filter_stride, padding="VALID")
                z = tf.reshape(conv, [num_inputs, -1]) + b
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
