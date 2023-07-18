#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include "config.c"
#include <time.h>

typedef struct
{
    double input[INPUT_SIZE];
    double output[OUTPUT_SIZE];
} Data;

typedef struct
{
    double weights1[INPUT_SIZE][HIDDEN_SIZE1];
    double biases1[HIDDEN_SIZE1];
    double weights2[HIDDEN_SIZE1][HIDDEN_SIZE2];
    double biases2[HIDDEN_SIZE2];
    double weights3[HIDDEN_SIZE2][OUTPUT_SIZE];
    double biases3[OUTPUT_SIZE];
} Model;

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

void load_data(Data *data, const char *filename)
{

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    unsigned char buffer[INPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; ++i)
    {
        if (fread(buffer, sizeof(buffer), 1, file) != 1)
        {
            printf("Failed to read data from file: %s\n", filename);
            exit(1);
        }
        data->input[i] = (double)buffer[i] / 255.0;
    }
    fclose(file);
}

void load_labels(Data *data, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }

    unsigned char label;
    if (fread(&label, sizeof(label), 1, file) != 1)
    {
        printf("Failed to read label from file: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        data->output[i] = (i == label) ? 1.0 : 0.0;
    }

    fclose(file);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

void forward_propagation(Data *data, Model *model, double *loss, double *accuracy)
{
    // Forward propagation
    double hidden1[HIDDEN_SIZE1];
    double hidden2[HIDDEN_SIZE2];
    double output[OUTPUT_SIZE];

    // First hidden layer
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            sum += data->input[j] * model->weights1[j][i];
        }
        hidden1[i] = sigmoid(sum + model->biases1[i]);
    }

    // Second hidden layer
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE1; ++j)
        {
            sum += hidden1[j] * model->weights2[j][i];
        }
        hidden2[i] = sigmoid(sum + model->biases2[i]);
    }

    // Output layer
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            sum += hidden2[j] * model->weights3[j][i];
        }
        output[i] = sigmoid(sum + model->biases3[i]);
    }

    // Calculate loss and accuracy
    double max_output = output[0];
    int predicted_label = 0;
    int target_label_found = 0;

    //#pragma omp parallel for reduction(max : max_output)
    for (int i = 1; i < OUTPUT_SIZE; ++i)
    {
        if (output[i] > max_output)
        {
            //#pragma omp critical
            {
                if (output[i] > max_output)
                {
                    max_output = output[i];
                    predicted_label = i;
                }
            }
        }
    }

    double target_label = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        if (data->output[i] == 1.0)
        {
            #pragma omp critical
            {
                if (!target_label_found)
                {
                    target_label = i;
                    target_label_found = 1;
                }
            }
        }
    }

    #pragma omp atomic
    *loss += -log(output[(int)target_label]);

    #pragma omp atomic
    *accuracy += (predicted_label == target_label) ? 1.0 : 0.0;
}


void reforward_propagation(Data *data, Model *model, double *loss, double *accuracy)
{
    // Forward propagation
    double hidden1[HIDDEN_SIZE1];
    double hidden2[HIDDEN_SIZE2];
    double output[OUTPUT_SIZE];

    // First hidden layer
    int i, j;
#pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {

      double sum = 0.0;
//#pragma omp parallel for
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            sum += data->input[j] * model->weights1[j][i];
        }
        hidden1[i] = sigmoid(sum + model->biases1[i]);
    }

    // Second hidden layer
#pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        double sum = 0.0;
//#pragma omp parallel for  
      for (int j = 0; j < HIDDEN_SIZE1; ++j)
        {
            sum += hidden1[j] * model->weights2[j][i];
        }
        hidden2[i] = sigmoid(sum + model->biases2[i]);
    }

    // Output layer
#pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        double sum = 0.0;
//#pragma omp parallel for  
      for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            sum += hidden2[j] * model->weights3[j][i];
        }
        output[i] = sigmoid(sum + model->biases3[i]);
    }

    // Calculate loss and accuracy
    double max_output = output[0];
    int predicted_label = 0;
    for (int i = 1; i < OUTPUT_SIZE; ++i)
    {
        if (output[i] > max_output)
        {
            max_output = output[i];
            predicted_label = i;
        }
    }

    double target_label = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        if (data->output[i] == 1.0)
        {
            target_label = i;
            break;
        }
    }

    *loss += -log(output[(int)target_label]);
    *accuracy += (predicted_label == target_label) ? 1.0 : 0.0;
}

void backward_propagation(Data *data, Model *model)
{
    // Backward propagation
    double hidden1[HIDDEN_SIZE1];
    double hidden2[HIDDEN_SIZE2];
    double output[OUTPUT_SIZE];

    // First hidden layer
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {
        double sum = 0.0;

        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            sum += data->input[j] * model->weights1[j][i];
        }
        hidden1[i] = sigmoid(sum + model->biases1[i]);
    }

    // Second hidden layer
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE1; ++j)
        {
            sum += hidden1[j] * model->weights2[j][i];
        }
        hidden2[i] = sigmoid(sum + model->biases2[i]);
    }

    // Output layer
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            sum += hidden2[j] * model->weights3[j][i];
        }
        output[i] = sigmoid(sum + model->biases3[i]);
    }

    // Calculate gradients and update weights/biases
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        double gradient = (output[i] - data->output[i]) * sigmoid_derivative(output[i]);
        for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            model->weights3[j][i] -= LEARNING_RATE * gradient * hidden2[j];
            model->biases3[i] -= LEARNING_RATE * gradient;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        double gradient = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; ++j)
        {
            gradient += (output[j] - data->output[j]) * sigmoid_derivative(output[j]) * model->weights3[i][j];
        }

        for (int j = 0; j < HIDDEN_SIZE1; ++j)
        {
            model->weights2[j][i] -= LEARNING_RATE * gradient * hidden1[j];
            model->biases2[i] -= LEARNING_RATE * gradient;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {
        double gradient = 0.0;
        for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            gradient += (output[j] - data->output[j]) * sigmoid_derivative(output[j]) * model->weights3[j][i];
        }

        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            model->weights1[j][i] -= LEARNING_RATE * gradient * data->input[j];
            model->biases1[i] -= LEARNING_RATE * gradient;
        }
    }
}



void train_model(Model *model, Data *training_data, Data *testing_data)
{
double start; 
double end; 
  for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
start = omp_get_wtime(); 
        double total_loss = 0.0;
        double total_accuracy = 0.0;

        // Training

        for (int i = 0; i < TRAINING_SIZE; ++i)
        {
            forward_propagation(&training_data[i], model, &total_loss, &total_accuracy);
            backward_propagation(&training_data[i], model);
        }

        double avg_loss = total_loss / TRAINING_SIZE;
        double avg_accuracy = total_accuracy / TRAINING_SIZE;

end = omp_get_wtime(); 

        printf("Epoch %d - Training Loss: %f, Training Accuracy: %f, Time: %.2f\n", epoch + 1, avg_loss, avg_accuracy,end-start);

        total_loss = 0.0;
        total_accuracy = 0.0;

        // Testing
        for (int i = 0; i < TESTING_SIZE; ++i)
        {
            forward_propagation(&testing_data[i], model, &total_loss, &total_accuracy);
        }

        avg_loss = total_loss / TESTING_SIZE;
        avg_accuracy = total_accuracy / TESTING_SIZE;

        printf("Epoch %d - Testing Loss: %f, Testing Accuracy: %f\n", epoch + 1, avg_loss, avg_accuracy);
    }
}

int main()
{
omp_set_num_threads(8);
int n_th = omp_thread_count();
printf("Nb thread = %d\n",n_th);
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    // Load data
    Data *training_data = malloc(TRAINING_SIZE * sizeof(Data));
    Data *testing_data = malloc(TESTING_SIZE * sizeof(Data));
    clock_t start1, end1;
    double cpu_time_used1;
    start1 = clock();
#pragma omp parallel for schedule(dynamic, 500)
    for (int i = 0; i < TRAINING_SIZE; ++i)
    {
        char filename[30];
        sprintf(filename, "data/train-images.idx3-ubyte", i + 1);
        load_data(&training_data[i], filename);

        sprintf(filename, "data/train-labels.idx1-ubyte", i + 1);
        load_labels(&training_data[i], filename);
    }
    end1 = clock();

    cpu_time_used1 = ((double)(end1 - start1)) / (CLOCKS_PER_SEC * n_th);

    printf("Load train data in %.2f seconds\n", cpu_time_used1);

 clock_t start3, end3;
    double cpu_time_used3;
    start3 = clock();
    Model *model = malloc(sizeof(Model));

    for (int i = 0; i < INPUT_SIZE; ++i)
    {
        for (int j = 0; j < HIDDEN_SIZE1; ++j)
        {
            model->weights1[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {
        model->biases1[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < HIDDEN_SIZE1; ++i)
    {
        for (int j = 0; j < HIDDEN_SIZE2; ++j)
        {
            model->weights2[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        model->biases2[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < HIDDEN_SIZE2; ++i)
    {
        for (int j = 0; j < OUTPUT_SIZE; ++j)
        {
            model->weights3[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        model->biases3[i] = (double)rand() / RAND_MAX;
    }

end3 = clock();

    cpu_time_used3 = ((double)(end3 - start3))  / (CLOCKS_PER_SEC * n_th);

    printf("Model initizalized in %.2f seconds\n", cpu_time_used3);

 clock_t start4, end4;
    double cpu_time_used4;
    start4 = clock();
    // Train model
    printf("Start training...\n");
    train_model(model, training_data, testing_data);
end4 = clock();

    cpu_time_used4 = ((double)(end4 - start4))  / (CLOCKS_PER_SEC * n_th);

    printf("Model trained in %.2f seconds\n", cpu_time_used4);
    // Cleanup
    free(training_data);
    free(testing_data);
    free(model);

    end = clock();

    cpu_time_used = ((double)(end - start))  / (CLOCKS_PER_SEC * n_th);

    printf("Execution time: %.2f seconds\n", cpu_time_used);

    return 0;
}
