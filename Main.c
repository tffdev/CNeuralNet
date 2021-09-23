#include <stdio.h>
#include <stdarg.h>

#include "NeuralNetwork.h"

Input NewInput(int size, ...)
{
    va_list ptr;
    va_start(ptr, size);

    Input input = { .m_Size = 0, .m_Values = {0} };
    input.m_Size = size;

    for (int i = 0; i < size; i++) {
        input.m_Values[i] = va_arg(ptr, NodeValue);
    }

    va_end(ptr);
    return input;
}

void PrintNetwork(Network* network) {
    for (size_t layerIndex = 0; layerIndex < network->m_NumLayers; layerIndex++) {
        for (size_t nodeIndex = 0; nodeIndex < network->m_NodeLayers[layerIndex].m_Size; nodeIndex++) {
            Node node = network->m_NodeLayers[layerIndex].m_Nodes[nodeIndex];
            printf("\tNode %llu %llu: %.4lf\n", layerIndex, nodeIndex, node.m_Value);
            // Print synapses
            for (size_t synIndex = 0; synIndex < node.m_SynapseCount; synIndex++) {
                printf("\t\tSynapse %llu: %.4lf\n", synIndex, node.m_Synapses[synIndex].m_Weight);
            }
        }
    }
}

void PrintResult(Result* result) {
    printf("Result:\n");
    for (size_t i = 0; i < result->m_Size; i++) {
        printf("\t%llu: %.3lf\n", i, result->m_Values[i]);
    }
}

int main() {
    NetworkSettings settings = {
        .m_Size = 3,
        .m_LayerSizes = {2,2,1},
        .m_LearningRate = 0.01,
        .m_Momentum = 0.1,
        .m_ActivationFunction = ACTIVATION_FUNCTION_RELU
    };
    Network* network = CreateNetwork(&settings);

    /*
     * I1->A = 1.0
     * I2->A = -1.0
     * I1->B = -1.0
     * I2->B = 1.0
     * A->O = 1.0
     * B->O = 1.0
     *
     *   I1 - A \
     *      X    O
     *   I2 - B /
     */

     // Manually assign the network's synapse weights to make a simple XOR network
     // Node A
    SetSynapseWeight(network, 1, 0, 0, 1.0);
    SetSynapseWeight(network, 1, 0, 1, -1.0);

    // Node B
    SetSynapseWeight(network, 1, 1, 0, -1.0);
    SetSynapseWeight(network, 1, 1, 1, 1.0);

    // Final output node
    SetSynapseWeight(network, 2, 0, 0, 1.0);
    SetSynapseWeight(network, 2, 0, 1, 1.0);

    Input input = NewInput(2, 1.0, 0.0);
    Result result = ForwardPropagate(network, input);
    PrintNetwork(network);
    PrintResult(&result);

    FreeNetwork(network);
}